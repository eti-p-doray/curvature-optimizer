import tensorflow_datasets as tfds  # TFDS for MNIST
import tensorflow as tf             # TensorFlow operations
from typing import Callable, Any, Protocol, Tuple, NamedTuple
from dataclasses import dataclass

from flax import linen as nn  # Linen API

from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import core
from flax import struct
import optax                           # Common loss functions and optimizers
import chex

import jax
import jax.numpy as jnp  # JAX NumPy

jax.config.update("jax_traceback_filtering", 'off')
#jax.config.update("jax_debug_nans", True)

import optimizers
import training

def get_image_datasets(name, batch_size):
  """Load MNIST train and test datasets into memory."""
  (train_ds, test_ds) = tfds.load(name, split=['train', 'test'], as_supervised=True)

  def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., tf.cast(label, tf.int32)

  train_ds = train_ds.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  test_ds = test_ds.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

  train_ds = train_ds.shuffle(1024) # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
  train_ds = train_ds.batch(batch_size, drop_remainder=True) # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
  train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

  test_ds = test_ds.shuffle(1024) # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
  test_ds = test_ds.batch(batch_size, drop_remainder=True) # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
  test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

  return train_ds, test_ds

class CNN(nn.Module):
  """A simple CNN model."""
  num_classes: int

  def setup(self):
    # Submodule names are derived by the attributes you assign to. In this
    # case, "dense1" and "dense2". This follows the logic in PyTorch.
    self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=jax.nn.initializers.glorot_normal())
    self.conv2 = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=jax.nn.initializers.glorot_normal())
    self.dense1 = nn.Dense(features=256, kernel_init=jax.nn.initializers.glorot_normal())
    self.dense2 = nn.Dense(features=self.num_classes, kernel_init=jax.nn.initializers.glorot_normal())

  def __call__(self, x):
    x = self.conv1(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = self.conv2(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = self.dense1(x)
    x = nn.relu(x)
    x = self.dense2(x)
    return x

class VGG16(nn.Module):
  num_classes: int
  dropout_rate: float = 0.2
  output: str='linear'
  dtype: str='float32'

  class ConvBlock(nn.Module):
    features: int
    num_layers: int
    dtype: str

    def setup(self):
      layers = []
      for l in range(self.num_layers):
        layers.append(nn.Conv(features=self.features, kernel_size=(3, 3), padding='same', dtype=self.dtype, kernel_init=jax.nn.initializers.glorot_normal()))
      self.layers = layers

    def __call__(self, x):
      for l in self.layers:
        x = l(x)
        x = nn.relu(x)
      x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
      return x

  def setup(self):
    self.conv1 = self.ConvBlock(features=32, num_layers=2, dtype=self.dtype)
    self.conv2 = self.ConvBlock(features=64, num_layers=2, dtype=self.dtype)
    self.conv3 = self.ConvBlock(features=128, num_layers=2, dtype=self.dtype)
    self.dense1 = nn.Dense(features=128, kernel_init=jax.nn.initializers.glorot_normal())
    self.dense2 = nn.Dense(features=self.num_classes, kernel_init=jax.nn.initializers.glorot_normal())

      
  @nn.compact
  def __call__(self, x, training=False):
    if self.output not in ['softmax', 'log_softmax', 'sigmoid', 'linear', 'log_sigmoid']:
        raise ValueError('Wrong argument. Possible choices for output are "softmax", "sigmoid", "log_sigmoid", "linear".')
    
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)

    x = x.reshape((x.shape[0], -1))  # flatten
    
    # Fully conected
    #x = jnp.mean(x, axis=(2, 3))
    x = self.dense1(x)
    x = nn.relu(x)
    #x = nn.BatchNorm()(x, use_running_average=not training)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training) 
    return self.dense2(x)

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')
  
def crossentrop_loss(y_pred, y):
  return optax.softmax_cross_entropy_with_integer_labels(logits=y_pred, labels=y).sum()

def create_img_train_state(module, dims, rng, state_class, tx):
  """Creates an initial `TrainState`."""
  params = module.init(rng, jnp.empty(dims))['params'] # initialize parameters by passing a template image
  opt_state = tx.init(params)

  return state_class(
      apply_fn=module.apply, params=params, tx=tx,
      opt_state=opt_state,
      loss_fn=crossentrop_loss,
      loss_hessian_fn=optimizers.sample_crossentropy_hessian,
      rng_key=rng,
      initial_metrics=Metrics)

num_epochs = 2
batch_size = 32


tf.random.set_seed(0)

#module = CNN(10)
module = VGG16(10)
rng = jax.random.PRNGKey(0)
dims = [1, 32, 32, 3]
print(module.tabulate(rng, jnp.ones(dims)))

tx = optimizers.kalman_blockwise_spectral_transformation(1.0, 1.0, 16, jax.random.PRNGKey(0))
#tx = optimizers.kalman_blockwise_trace_transformation(1.0, 1.0)
#tx = optax.sgd(0.01 / batch_size, 0.9)

train_ds, test_ds = get_image_datasets('cifar10', batch_size)
state = create_img_train_state(module, dims, jax.random.PRNGKey(1), training.NaturalTrainState, tx)
#state = create_img_train_state(module, dims, jax.random.PRNGKey(1), training.TrainState, tx)

#with jax.profiler.trace("./jax-trace", create_perfetto_trace=True):
state = training.train(train_ds, test_ds, state, num_epochs)
jax.block_until_ready(state)