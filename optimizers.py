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

def kalman_blockwise_trace_transformation(fading: float, lr: float) -> training.NaturalGradientTransformation:
  @flax.struct.dataclass
  class State:
    fim_trace: chex.ArrayTree

  def init_fim_trace(param: jax.Array):
    variance = jnp.var(param)
    if (variance > 0):
      return 1.0 / variance
    fan_in, fan_out = compute_fans(param.shape)
    return fan_in

  def init(params: chex.ArrayTree):
    fim_trace = jax.tree_util.tree_map(
        init_fim_trace, params)
    return State(fim_trace=fim_trace)

  def transform_update(updates, state: State, params=None):
    updates = jax.tree_util.tree_map(
        lambda u, information: -u / information * lr, updates, state.fim_trace)
    fim_trace = jax.tree_util.tree_map(
        lambda i: i * fading, state.fim_trace)
    return updates, state.replace(fim_trace=fim_trace)

  def consume_sample(information_samples, state: State, params=None):
    def consume_sample_block(sample, fim_trace, param):
      size = float(jnp.size(param))
      return fim_trace + lr * jnp.sum(jnp.multiply(sample, sample)) / size
    fim_trace = jax.tree_util.tree_map(
        consume_sample_block, information_samples, state.fim_trace, params)
    return state.replace(fim_trace=fim_trace)
  
  @jax.jit
  def update_step(state: State, params=None):
    return state
  
  return training.NaturalGradientTransformation(init, transform_update, consume_sample, update_step)

def kalman_blockwise_spectral_transformation(fading: float, lr: float, max_rank: int, rng) -> training.NaturalGradientTransformation:
  @flax.struct.dataclass
  class InformationState:
    trace: float
    basis: jax.Array
    samples: jax.Array
    rank: int
    kernel: jax.Array
  
  @flax.struct.dataclass
  class State:
    fim: chex.ArrayTree
    rng_key: jax.random.KeyArray
    rank: int

  def init_fim_trace(param: jax.Array):
    variance = jnp.var(param)
    if (variance > 0):
      return 1.0 / variance
    fan_in, fan_out = compute_fans(param.shape)
    return fan_in

  def init(params: chex.ArrayTree):
    def init_block(param):
      trace = init_fim_trace(param)
      basis = jnp.zeros((max_rank,) + param.shape)
      samples = jnp.zeros((max_rank,) + param.shape)
      kernel = jnp.zeros([max_rank])
      return InformationState(trace=trace, basis=basis, samples=samples, kernel=kernel, rank=0)
    return State(fim=jax.tree_util.tree_map(init_block, params), rng_key=rng, rank=0)

  def augment_samples_block(sample: jax.Array, fim: InformationState) -> InformationState:
    samples = fim.samples.at[fim.rank,...].set(sample)
    return fim.replace(samples=samples,rank=fim.rank+1)

  """def compress_first_samples_block(param: jax.Array, fim: InformationState, rng_key: jax.random.KeyArray) -> InformationState:
    sum_dims = list(range(1, fim.basis.ndim))
  
    basis = fim.samples
    kernel = jnp.tensordot(basis, basis, [sum_dims,sum_dims])

    s, v = jnp.linalg.eigh(kernel)
    s = jnp.maximum(s, 0)
    return fim.replace(rank=0, kernel=s, basis=basis)"""

  def compress_samples_block(param: jax.Array, fim: InformationState, rng_key: jax.random.KeyArray) -> InformationState:
    rank = max_rank
    sum_dims = list(range(1, fim.basis.ndim))
    # maybe jnp.tensordot(transform.T, fim.basis, axes=[[0],[0]])
    #B = jnp.tensordot(fim.basis, fim.samples, [sum_dims,sum_dims])
    #D = jnp.tensordot(fim.samples, fim.samples, [sum_dims,sum_dims])
    #kernel = jnp.concatenate([
    #  jnp.concatenate([jnp.diag(fim.kernel), B.T], 0), 
    #  jnp.concatenate([B, D], 0)], 1)
    basis = jnp.concatenate([fim.basis, fim.samples], 0)
    kernel = jnp.tensordot(basis, basis, [sum_dims,sum_dims])

    s, v = jnp.linalg.eigh(kernel)
    s = jnp.maximum(s, 0.0)
    size = float(jnp.size(param))
    kernel_spill = jnp.sum(s[:-rank]) / size

    #transform = jax.random.orthogonal(rng_key, rank)
    #u = jnp.dot(v[:,-rank:], transform.T)
    basis = jnp.tensordot(v[:,-rank:], basis, axes=[[0],[0]])
    kernel = s[-rank:]

    #print(fim.trace, kernel_spill)
    
    #kernel = jnp.zeros(fim.kernel.shape).at[:rank,:rank].set(
    #  jnp.dot(transform * s[-rank:], transform.T))

    return fim.replace(rank=0, kernel=kernel, basis=basis, trace=fim.trace+kernel_spill)

  def transform_update_block(update, fim: InformationState):
    X = jnp.tensordot(fim.basis, update, update.ndim)
    X = X / (fim.kernel + fim.trace)
    #X = jax.scipy.linalg.solve(fim.kernel + fim.trace * jnp.eye(fim.kernel.shape[0]), X, assume_a='pos')
    return (jnp.tensordot(fim.basis, X, [[0], [0]]) - update)  * lr / fim.trace
  
  def transform_update(updates, state: State, params=None):
    updates = jax.tree_util.tree_map(
        transform_update_block, updates, state.fim)
    return updates, state
  
  @jax.jit
  def augment_samples(information_samples, state: State):
    fim = jax.tree_util.tree_map(
        augment_samples_block, information_samples, state.fim)
    return state.replace(fim=fim, rank=state.rank+1)
  
  @jax.jit
  def compress_samples(state: State, params):
    rng_key, subkey = jax.random.split(state.rng_key)
    treedef = jax.tree_structure(params)
    subkeys = jax.random.split(subkey, treedef.num_leaves)
    subkeys = jax.tree_unflatten(treedef, subkeys)
    
    fim = jax.tree_util.tree_map(
        compress_samples_block, params, state.fim, subkeys)
    return state.replace(fim=fim, rng_key=rng_key, rank=0)


  def consume_sample(information_samples, state: State, params=None):
    return augment_samples(information_samples, state)
  
  def update_step(state: State, params=None):
    if state.rank == max_rank:
      state = compress_samples(state, params)
    return state
  
  return training.NaturalGradientTransformation(init, transform_update, consume_sample, update_step)