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

import optimizers
import training

def get_mnist_datasets(batch_size):
  """Load MNIST train and test datasets into memory."""
  (train_ds, test_ds) = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

  def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

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

  def setup(self):
    # Submodule names are derived by the attributes you assign to. In this
    # case, "dense1" and "dense2". This follows the logic in PyTorch.
    self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=jax.nn.initializers.glorot_normal())
    self.conv2 = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=jax.nn.initializers.glorot_normal())
    self.dense1 = nn.Dense(features=256, kernel_init=jax.nn.initializers.glorot_normal())
    self.dense2 = nn.Dense(features=10, kernel_init=jax.nn.initializers.glorot_normal())

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

tf.random.set_seed(0)

cnn = CNN()
rng = jax.random.PRNGKey(0)
print(cnn.tabulate(rng, jnp.ones((1, 28, 28, 1))))

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')
  
def crossentrop_loss(y_pred, y):
  return optax.softmax_cross_entropy_with_integer_labels(logits=y_pred, labels=y).sum()

def create_mnist_train_state(module, rng, state_class, tx):
  """Creates an initial `TrainState`."""
  params = module.init(rng, jnp.empty([1, 28, 28, 1]))['params'] # initialize parameters by passing a template image
  opt_state = tx.init(params)

  return state_class(
      apply_fn=module.apply, params=params, tx=tx,
      opt_state=opt_state,
      loss_fn=crossentrop_loss,
      loss_hessian_fn=optimizers.sample_crossentropy_hessian,
      rng_key=rng,
      initial_metrics=Metrics)

num_epochs = 10
batch_size = 32

train_ds, test_ds = get_mnist_datasets(batch_size)

learning_rate = 0.01 / 32
momentum = 0.9

tx = optimizers.basic_kalman_trace_transformation(1.0, 0.5)
state = create_mnist_train_state(cnn, jax.random.PRNGKey(0), training.NaturalTrainState, tx)
#state = create_mnist_train_state(cnn, jax.random.PRNGKey(0), TrainState, optax.sgd(learning_rate, momentum))

training.train(train_ds, test_ds, state, num_epochs)