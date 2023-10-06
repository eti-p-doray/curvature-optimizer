from typing import Callable, Any, Protocol, Tuple
from dataclasses import dataclass

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

  @jax.jit
  def train_step(state, metrics, batch):
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

def train(train_ds, test_ds, state, epochs):
  for epoch in range(epochs):
    metrics = state.initial_metrics.empty()
    for batch in train_ds.as_numpy_iterator():
      # Run optimization steps over training batches and compute batch metrics
      state, metrics = state.train_step(metrics, batch) # get updated train state (which contains the updated parameters)
    for metric,value in metrics.compute().items():
      print(f'Train {metric}: {value}')

    metrics = state.initial_metrics.empty()
    for batch in test_ds.as_numpy_iterator():
      metrics = state.eval_step(metrics, batch) # aggregate batch metrics
    for metric,value in metrics.compute().items():
      print(f'Test {metric}: {value}')