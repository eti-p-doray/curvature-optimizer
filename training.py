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

def compute_metrics(metrics, *, loss, logits, labels):
  metric_updates = metrics.single_from_model_output(
    logits=logits, labels=labels, loss=loss)
  return metrics.merge(metric_updates)

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

class UpdateStepFn(Protocol):
  def __call__(self,
      state: chex.ArrayTree,
      params: chex.ArrayTree) -> chex.ArrayTree:
    ...


@dataclass
class NaturalGradientTransformation:
  init: TransformInitFn
  transform_update: TransformUpdateFn
  consume_sample: ConsumeSampleFn
  update_step: UpdateStepFn

@flax.struct.dataclass
class TrainState:
  apply_fn: Callable = flax.struct.field(pytree_node=False)
  tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
  loss_fn: Callable = flax.struct.field(pytree_node=False)
  loss_hessian_fn: Callable = flax.struct.field(pytree_node=False)

  initial_metrics: clu.metrics.Collection = flax.struct.field(pytree_node=False)
  params: chex.ArrayTree
  opt_state: optax.OptState
  rng_key: jax.random.KeyArray

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

@flax.struct.dataclass
class NaturalTrainState(TrainState):
  tx: NaturalGradientTransformation = flax.struct.field(pytree_node=False)

  """@jax.jit
  def train_step(state, metrics, batch):
    new_key, subkey = jax.random.split(state.rng_key)
    x, y = batch

    def predict_and_loss_fn(params):
      logits = state.apply_fn({'params': params}, x)
      loss = state.loss_fn(logits, y)
      return loss, logits
    
    with jax.profiler.TraceAnnotation("value_and_grad"):
      grad_fn = jax.value_and_grad(predict_and_loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    metrics = compute_metrics(metrics, loss=loss, logits=logits, labels=y)

    def backward_sample(x, rng_key):
      def predict(params):
        return state.apply_fn({'params': params}, jnp.expand_dims(x, 0))
      logits, vjp_fun = jax.vjp(predict, state.params)
      sample = state.loss_hessian_fn(logits, jax.random.normal(rng_key, logits.shape))
      return vjp_fun(sample)[0]
    
    with jax.profiler.TraceAnnotation("backward_sample"):
      #information_samples = jax.vmap(backward_sample)(x, loss_hessian_samples)
      information_samples = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x,0), backward_sample(x[0,...], subkey))

    updates, new_opt_state = state.tx.transform_update(grads, state.opt_state, state.params)

    new_opt_state = state.tx.consume_sample(information_samples, new_opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    return state.replace(
        params=new_params,
        opt_state=new_opt_state,
        rng_key=new_key
    ), metrics"""
  
  @jax.profiler.annotate_function
  @jax.jit
  def train_step_impl(state, metrics, batch):
    new_key, subkey = jax.random.split(state.rng_key)
    x, y = batch

    def predict(params):
      return state.apply_fn({'params': params}, x)
    def loss(logits):
      return state.loss_fn(logits, y)
    
    logits, grad_fn = jax.vjp(predict, state.params)
    loss_grad_fn = jax.value_and_grad(loss)
    loss, loss_grad = loss_grad_fn(logits)
    grads = grad_fn(loss_grad)[0]

    samples = jax.random.normal(subkey, logits.shape)
    samples = jax.vmap(state.loss_hessian_fn)(logits, samples)
    samples = grad_fn(samples)[0]

    metrics = compute_metrics(metrics, loss=loss, logits=logits, labels=y)
    updates, new_opt_state = state.tx.transform_update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_opt_state = state.tx.consume_sample(samples, new_opt_state, state.params)
    return state.replace(
        params=new_params,
        opt_state=new_opt_state,
        rng_key=new_key
    ), metrics
  
  def train_step(state, metrics, batch):
    state, metrics = state.train_step_impl(metrics, batch)
    new_opt_state = state.tx.update_step(state.opt_state, state.params)
    return state.replace(opt_state=new_opt_state), metrics

def train(train_ds, test_ds, state, epochs):
  for epoch in range(epochs):
    metrics = state.initial_metrics.empty()
    
    #if epoch > 0:
      #jax.profiler.start_trace("./jax-trace", create_perfetto_trace=True)
    
    start = perf_counter()
    with Bar('Training', max=train_ds.cardinality().numpy()) as bar:
      for i, batch in enumerate(train_ds.as_numpy_iterator()):
        # Run optimization steps over training batches and compute batch metrics
        state, metrics = state.train_step(metrics, batch) # get updated train state (which contains the updated parameters)
        bar.next()
    jax.block_until_ready(state)
    print("Elapsed time:", perf_counter() - start) 
    for metric,value in metrics.compute().items():
      print(f'Train {metric}: {value}')

    metrics = state.initial_metrics.empty()
    for batch in test_ds.as_numpy_iterator():
      metrics = state.eval_step(metrics, batch) # aggregate batch metrics
    for metric,value in metrics.compute().items():
      print(f'Test {metric}: {value}')

  jax.block_until_ready(state)
  #jax.profiler.stop_trace()
  return state