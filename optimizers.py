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
  return z * samples - y * jnp.sum(z * samples, axis=-1, keepdims=True)

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
  
  return training.NaturalGradientTransformation(init, transform_update, consume_sample)

def kalman_blockwise_spectral_transformation(fading: float, lr: float, kernel_rank: int, buffer_rank: int, rng) -> training.NaturalGradientTransformation:
  @flax.struct.dataclass
  class InformationState:
    kernel_trace: float
    samples_trace: float
    basis: jax.Array
    samples: jax.Array
    rank: int
    kernel: jax.Array
  
  @flax.struct.dataclass
  class State:
    fim: chex.ArrayTree
    rng_key: jax.random.PRNGKey
    rank: int

  def init_fim_trace(param: jax.Array):
    variance = jnp.var(param)
    if (variance > 0):
      return float(1.0 / variance)
    fan_in, fan_out = compute_fans(param.shape)
    return float(fan_in)

  def init(params: chex.ArrayTree):
    def init_block(param):
      kernel_trace = init_fim_trace(param)
      basis = jnp.zeros((kernel_rank,) + param.shape, dtype='float32')
      samples = jnp.zeros((buffer_rank,) + param.shape, dtype='float32')
      kernel = jnp.zeros([kernel_rank], dtype='float32')
      return InformationState(kernel_trace=kernel_trace, samples_trace=0.0, basis=basis, samples=samples, kernel=kernel, rank=0)
    return State(fim=jax.tree_util.tree_map(init_block, params), rng_key=rng, rank=0)

  def augment_samples_block(sample: jax.Array, fim: InformationState) -> InformationState:
    samples = fim.samples.at[fim.rank,...].set(sample)
    trace = jnp.tensordot(sample, sample, sample.ndim)
    return fim.replace(samples=samples, samples_trace=fim.samples_trace+trace, rank=fim.rank+1)

  @jax.jit
  def compress_samples_block(param: jax.Array, fim: InformationState) -> InformationState:
    rank = kernel_rank
    sum_dims = list(range(1, fim.basis.ndim))
    # maybe jnp.tensordot(transform.T, fim.basis, axes=[[0],[0]])
    
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

    return fim.replace(rank=0, 
                       kernel=kernel, 
                       basis=basis, 
                       kernel_trace=fim.kernel_trace+kernel_spill, 
                       samples_trace=0.0)

  @jax.jit
  def transform_update_block(update, fim: InformationState):
    X = jnp.tensordot(fim.basis, update, update.ndim)
    size = float(jnp.size(update))
    trace = fim.kernel_trace + fim.samples_trace / size
    X = X / (fim.kernel + trace)
    #X = jax.scipy.linalg.solve(fim.kernel + fim.trace * jnp.eye(fim.kernel.shape[0]), X, assume_a='pos')
    return (jnp.tensordot(fim.basis, X, [[0], [0]]) - update)  * lr / trace
  
  def transform_update(updates, state: State, params=None):
    updates = jax.tree_util.tree_map(
        transform_update_block, updates, state.fim)
    state = jax.lax.cond(state.rank == buffer_rank, 
                        lambda s, p: compress_samples(s, p), lambda s, p: s,
                        state, params)
    return updates, state
  
  @jax.jit
  def augment_samples(information_samples, state: State):
    fim = jax.tree_util.tree_map(
        augment_samples_block, information_samples, state.fim)
    return state.replace(fim=fim, rank=state.rank+1)
  
  @jax.jit
  def compress_samples(state: State, params):
    #rng_key, subkey = jax.random.split(state.rng_key)
    #treedef = jax.tree_structure(params)
    #subkeys = jax.random.split(subkey, treedef.num_leaves)
    #subkeys = jax.tree_unflatten(treedef, subkeys)
    
    fim = jax.tree_util.tree_map(
        compress_samples_block, params, state.fim)
    return state.replace(fim=fim, rank=0)


  def consume_sample(information_samples, state: State, params=None):
    return augment_samples(information_samples, state)
  
  return training.NaturalGradientTransformation(init, transform_update, consume_sample)