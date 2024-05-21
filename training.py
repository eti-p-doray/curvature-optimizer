from typing import Callable, Any, Protocol, Tuple
from dataclasses import dataclass
from progress.bar import Bar
from time import perf_counter

import flax
import jax
import jax.numpy as jnp  # JAX NumPy
import optax
import chex
import clu
import time
import sys

class TransformInitFn(Protocol):
  def __call__(self, params: chex.ArrayTree) -> chex.ArrayTree:
    ...

class TransformUpdateFn(Protocol):
  def __call__(self,
      updates: chex.ArrayTree,
      state: chex.ArrayTree,
      params: chex.ArrayTree) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
    ...

class ConsumeSampleFn(Protocol):
  def __call__(self,
      information_samples: chex.ArrayTree,
      state: chex.ArrayTree,
      params: chex.ArrayTree) -> chex.ArrayTree:
    ...

@dataclass
class NaturalGradientTransformation:
  init: TransformInitFn
  transform_update: TransformUpdateFn
  consume_sample: ConsumeSampleFn

@flax.struct.dataclass
class TrainState:
  apply_fn: Callable = flax.struct.field(pytree_node=False)
  tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
  loss_fn: Callable = flax.struct.field(pytree_node=False)
  loss_hessian_fn: Callable = flax.struct.field(pytree_node=False)
  compute_metrics_fn: Callable = flax.struct.field(pytree_node=False)

  initial_metrics: clu.metrics.Collection = flax.struct.field(pytree_node=False)
  params: chex.ArrayTree
  opt_state: optax.OptState
  rng_key: jax.random.PRNGKey

  @jax.jit
  def train_step(state, metrics, batch):
    new_key, _ = jax.random.split(state.rng_key)
    x, y = batch
    """Train for a single step."""
    def predict_and_loss_fn(params):
      outputs = state.apply_fn({'params': params}, x)
      loss = state.loss_fn(outputs, y)
      return loss, outputs
    
    grad_fn = jax.value_and_grad(predict_and_loss_fn, has_aux=True)
    (loss, outputs), grads = grad_fn(state.params)
    metrics = state.compute_metrics_fn(metrics, loss=loss, outputs=outputs, labels=y)

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
    outputs = state.apply_fn({'params': state.params}, x)
    loss = state.loss_fn(outputs, y)
    return state.compute_metrics_fn(metrics, loss=loss, outputs=outputs, labels=y)

@flax.struct.dataclass
class NaturalTrainState(TrainState):
  tx: NaturalGradientTransformation = flax.struct.field(pytree_node=False)
  
  @jax.profiler.annotate_function
  @jax.jit
  def train_step_impl(state, metrics, batch):
    new_key, subkey = jax.random.split(state.rng_key)
    x, y = batch

    def predict(params):
      return state.apply_fn({'params': params}, x)
    def loss(outputs):
      return state.loss_fn(outputs, y) 
    
    outputs, grad_fn = jax.vjp(predict, state.params)
    loss_grad_fn = jax.value_and_grad(loss)
    loss, loss_grad = loss_grad_fn(outputs)
    grads = grad_fn(loss_grad)[0]

    # Map the random.normal across each component of the output 
    samples = jax.tree_map(lambda output: jax.random.normal(subkey, jnp.shape(output)), outputs)
    samples = state.loss_hessian_fn(outputs, samples)
    samples = grad_fn(samples)[0]

    metrics = state.compute_metrics_fn(metrics, loss=loss, outputs=outputs, labels=y)
    updates, new_opt_state = state.tx.transform_update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_opt_state = state.tx.consume_sample(samples, new_opt_state, state.params)
    return state.replace(
        params=new_params,
        opt_state=new_opt_state,
        rng_key=new_key
    ), metrics
  
  @jax.jit
  def train_step(state, metrics, batch):
    state, metrics = state.train_step_impl(metrics, batch)
    return state, metrics

def train(train_ds, state, cardinality):    
  metrics = state.initial_metrics.empty()
  start = perf_counter()
  with Bar('Training', max=cardinality) as bar:
    for batch in train_ds:
      # Run optimization steps over training batches and compute batch metrics
      state, metrics = state.train_step(metrics, batch) # get updated train state (which contains the updated parameters)
      bar.next()
  jax.block_until_ready(state)
  print("Elapsed time:", perf_counter() - start) 
  for metric,value in metrics.compute().items():
    print(f'Train {metric}: {value}')
  return state

def train_with_eval(train_ds, state, cardinality, test_ds=None, eval_every=100, test_batches=10, start_time=None):
    metrics = state.initial_metrics.empty()
    start = time.perf_counter()
    batch_count = 0
    
    with Bar('Training', max=cardinality) as bar:
        for batch in train_ds:
            state, metrics = state.train_step(metrics, batch)
            bar.next()
            batch_count += 1
            
            if batch_count % eval_every == 0:
                if start_time:
                    current_time = time.time()
                    elapsed_time = (current_time - start_time) / 60  # Time in minutes
                    print(f"\nElapsed Time: {elapsed_time:.2f} minutes")
                
                print(f"\nEvaluating on test set after {batch_count} batches...")
                test_metrics = eval_test_subset(test_ds, state, test_batches)
                print(f"Test Loss after {batch_count} batches: {test_metrics['loss']}")
                sys.stdout.flush()
    jax.block_until_ready(state)
    print("Elapsed time for training:", time.perf_counter() - start)
    for metric, value in metrics.compute().items():
        print(f'Train {metric}: {value}')
    return state


def eval_test_subset(test_ds, state, num_batches):
    metrics = state.initial_metrics.empty()
    batch_counter = 0
    for batch in test_ds:
        if batch_counter >= num_batches:
            break
        metrics = state.eval_step(metrics, batch)
        batch_counter += 1
    return metrics.compute()


def test(test_ds, state):
  metrics = state.initial_metrics.empty()
  for batch in test_ds:
    metrics = state.eval_step(metrics, batch) # aggregate batch metrics
  for metric,value in metrics.compute().items():
    print(f'Test {metric}: {value}')