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

def compute_fans(shape: Tuple[int,...]):
  """Computes the number of input and output units for a weight shape.
  Args:
    shape: Integer shape tuple or TF tensor shape.
  Returns:
    A tuple of integer scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)

class TransformInitFn(Protocol):
  def __call__(self, params: chex.ArrayTree) -> chex.ArrayTree:
    ...

class TransformUpdateFn(Protocol):
  def __call__(self,
      updates: chex.ArrayTree,
      state: chex.ArrayTree,
      params: chex.ArrayTree) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
    ...

class TransformInformationFn(Protocol):
  def __call__(self,
      information_samples: chex.ArrayTree,
      state: chex.ArrayTree,
      params: chex.ArrayTree) -> chex.ArrayTree:
    ...

@dataclass
class NaturalGradientTransformation:
  init: TransformInitFn
  transform_update: TransformUpdateFn
  transform_information: TransformInformationFn


def basic_kalman_trace_transformation(fading, lr) -> NaturalGradientTransformation:
  @struct.dataclass
  class State:
    fading: float
    lr: float
    information: chex.ArrayTree

  def init_information(param: jax.Array):
    variance = jnp.var(param)
    if (variance > 0):
      return 1.0 / variance
    fan_in, fan_out = compute_fans(param.shape)
    return fan_in

  def init(params: chex.ArrayTree):
    initializers = jax.tree_util.tree_map(
        init_information, params)
    return State(fading=fading, lr=lr, information=initializers)

  def transform_update(updates, state: State, params=None):
    updates = jax.tree_util.tree_map(
        lambda u, information: -u / information * state.lr, updates, state.information)
    information = jax.tree_util.tree_map(
        lambda i: i * state.fading, state.information)
    return updates, state.replace(information=information)

  def transform_information(information_samples, state: State, params=None):
    information = jax.tree_util.tree_map(
        lambda sample, information, param: information + state.lr * (2.0 - state.lr) * jnp.sum(jnp.multiply(sample, sample)) / float(jnp.size(param)), information_samples, state.information, params)
    return state.replace(information=information)
  
  return NaturalGradientTransformation(init, transform_update, transform_information)

def get_datasets(batch_size):
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

cnn = CNN()
print(cnn.tabulate(jax.random.key(0), jnp.ones((1, 28, 28, 1))))

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')

@struct.dataclass
class TrainState:
  apply_fn: Callable = struct.field(pytree_node=False)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  loss_fn: Callable = struct.field(pytree_node=False)
  loss_hessian_fn: Callable = struct.field(pytree_node=False)

  params: chex.ArrayTree
  opt_state: optax.OptState
  rng_key: jax.random.KeyArray
  
def crossentrop_loss(y_pred, y):
  return optax.softmax_cross_entropy_with_integer_labels(logits=y_pred, labels=y).sum()

def sample_crossentropy_hessian(predictions, samples):
  y = nn.activation.softmax(predictions)
  z = jnp.sqrt(y)
  return z * samples - y * jnp.sum(z * samples)

def create_mnist_train_state(module, rng, tx):
  """Creates an initial `TrainState`."""
  params = module.init(rng, jnp.empty([1, 28, 28, 1]))['params'] # initialize parameters by passing a template image
  opt_state = tx.init(params)

  return TrainState(
      apply_fn=module.apply, params=params, tx=tx,
      opt_state=opt_state,
      loss_fn=crossentrop_loss,
      loss_hessian_fn=sample_crossentropy_hessian,
      rng_key=rng)


def compute_metrics(metrics, *, loss, logits, labels):
  metric_updates = metrics.single_from_model_output(
    logits=logits, labels=labels, loss=loss)
  return metrics.merge(metric_updates)

@jax.jit
def natural_train_step(state, metrics, batch):
  new_key, subkey = jax.random.split(state.rng_key)
  x, y = batch

  """Train for a single step."""
  def predict_and_loss_fn(params):
    logits = state.apply_fn({'params': params}, x)
    loss = state.loss_fn(logits, y)
    return loss, logits
  
  grad_fn = jax.value_and_grad(predict_and_loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)

  metrics = compute_metrics(metrics, loss=loss, logits=logits, labels=y)

  def sample_loss_hessian(y_pred, rng_key):
    sample = jax.random.normal(rng_key, y_pred.shape)
    return jax.vmap(state.loss_hessian_fn)(y_pred, sample)

  loss_hessian_samples = sample_loss_hessian(logits, subkey)
  def predict(params):
    logits = state.apply_fn({'params': params}, x)
    return logits
  
  logits, vjp_fun = jax.vjp(predict, state.params)
  information_samples = vjp_fun(loss_hessian_samples)[0]

  updates, new_opt_state = state.tx.transform_update(grads, state.opt_state, state.params)

  new_opt_state = state.tx.transform_information(information_samples, new_opt_state, state.params)
  new_params = optax.apply_updates(state.params, updates)
  return state.replace(
      params=new_params,
      opt_state=new_opt_state,
      rng_key=new_key
  ), metrics

@jax.jit
def train_step(state, metrics, batch):
  new_key, _ = jax.random.split(state.rng_key)
  x, y = batch
  """Train for a single step."""
  def predict_and_loss_fn(params):
    logits = state.apply_fn({'params': params}, x)
    loss = state.loss_fn(logits, y)
    return loss, logits
  
  grad_fn = jax.value_and_grad(predict_and_loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  metrics = compute_metrics(metrics, loss=loss, logits=logits, labels=y)

  updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
  new_params = optax.apply_updates(state.params, updates)
  return state.replace(
      params=new_params,
      opt_state=new_opt_state,
      rng_key=new_key
  ), metrics

@jax.jit
def eval_step(state, metrics, batch):
  x, y = batch
  logits = state.apply_fn({'params': state.params}, x)
  loss = state.loss_fn(logits, y)
  return compute_metrics(metrics, loss=loss, logits=logits, labels=y)

def train_loop(train_ds, test_ds, train_step_fn, state, metrics, epochs):
  for epoch in range(epochs):
    metrics = metrics.empty()
    for batch in train_ds.as_numpy_iterator():
      # Run optimization steps over training batches and compute batch metrics
      state, metrics = train_step_fn(state, metrics, batch) # get updated train state (which contains the updated parameters)
    for metric,value in metrics.compute().items():
      print(f'Train {metric}: {value}')

    metrics = metrics.empty()
    for batch in test_ds.as_numpy_iterator():
      metrics = eval_step(state, metrics, batch) # aggregate batch metrics
    for metric,value in metrics.compute().items():
      print(f'Test {metric}: {value}')


num_epochs = 10
batch_size = 32

train_ds, test_ds = get_datasets(batch_size)

tf.random.set_seed(0)

learning_rate = 0.01 / 32
momentum = 0.9

#state = create_mnist_train_state(cnn, init_rng, basic_kalman_trace_transformation(1.0, 0.5))
state = create_mnist_train_state(cnn, jax.random.key(0), optax.sgd(learning_rate, momentum))

train_loop(train_ds, test_ds, train_step, state, Metrics.empty(), num_epochs)