from typing import Tuple

import flax
import jax
import jax.numpy as jnp  # JAX NumPy
import chex
from flax import linen as nn  # Linen API

import training

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

def sample_crossentropy_hessian(predictions, samples):
  y = nn.activation.softmax(predictions)
  z = jnp.sqrt(y)
  return z * samples - y * jnp.sum(z * samples)

def basic_kalman_trace_transformation(fading, lr) -> training.NaturalGradientTransformation:
  @flax.struct.dataclass
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
  
  return training.NaturalGradientTransformation(init, transform_update, transform_information)