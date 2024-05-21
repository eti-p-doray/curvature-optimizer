import time
import operator

import jax
import jax.numpy as jnp  # JAX NumPy

from flax import linen as nn  # Linen API

from clu import metrics
from flax import struct
import optax                           # Common loss functions and optimizers
import chex

jax.config.update("jax_traceback_filtering", 'off')
#jax.config.update("jax_debug_nans", True)

import optimizers
import training

def generate_polynomial_dataset(p, f, split, rng: jax.random.PRNGKey):
  a,b = jnp.mgrid[0:p, 0:p]
  y = f(a, b)
  n = y.size
  random_indices = jax.random.permutation(rng, n)
  train_indices = random_indices[:int(split*n)]
  test_indices = random_indices[train_indices.size:]
  x = jnp.reshape(jnp.stack((a, b), axis=-1), [-1, 2])
  y = jnp.ravel(y)
  return (x[train_indices,...], y[train_indices]), (x[test_indices,...], y[test_indices])

def batched(data, batch_size):
  x, y = data
  for i in range(0, y.size, batch_size):
    yield x[i:i+batch_size,...], y[i:i+batch_size,...]
  if i < y.size:
    yield x[i:,...], y[i:,...]

current_time_seed = int(time.time())
rng = jax.random.PRNGKey(current_time_seed)

p = 97
train_ds, test_ds = generate_polynomial_dataset(p, lambda a, b: jnp.mod(a + b, p), 0.9, rng)


@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')

class MLP(nn.Module):
  p: int
  
  @nn.compact
  def __call__(self, x):
    x = nn.Embed(num_embeddings=2*p, features=256, embedding_init=jax.nn.initializers.he_normal())(x)
    x = x.reshape((x.shape[0], -1)) # flatten
    #x = nn.Dense(features=512, kernel_init=jax.nn.initializers.he_normal())(x)
    x = nn.relu(x)
    # x = nn.Dense(features=194, kernel_init=jax.nn.initializers.he_normal())(x)
    # x = nn.relu(x)
    x = nn.Dense(features = p, kernel_init=jax.nn.initializers.he_normal())(x)
    return x
  
def crossentrop_loss(y_pred, y):
  return optax.softmax_cross_entropy_with_integer_labels(logits=y_pred, labels=y).sum()

def loss(y_pred_and_l2, y):
  return crossentrop_loss(y_pred_and_l2[0], y) + y_pred_and_l2[1] * 0.1

def apply(params, x):
  return (model.apply(params, x), jax.tree.reduce(operator.add, jax.tree_map(lambda param: jnp.sum(jnp.square(param)), params)))

def sample_hessian(prediction, sample):
  return (optimizers.sample_crossentropy_hessian(prediction[0], sample[0]), 0.0)

def compute_metrics(metrics, *, loss, outputs, labels):
  metric_updates = metrics.single_from_model_output(
    logits=outputs[0], labels=labels, loss=loss)
  return metrics.merge(metric_updates)

batch_size = 2328

model = MLP(p)
rng, subkey = jax.random.split(rng)
dummy_x = jnp.zeros(shape=(batch_size, 2), dtype=jnp.int32)
params = model.init(subkey, dummy_x)['params']

#tx = optax.adam(0.02)
tx = optimizers.kalman_blockwise_spectral_transformation(4.0, 1.0, 16, 48, jax.random.PRNGKey(1))
opt_state = tx.init(params)

rng, subkey = jax.random.split(rng)
state = training.NaturalTrainState(
      apply_fn=apply, params=params, tx=tx,
      opt_state=opt_state,
      loss_fn=loss,
      loss_hessian_fn=sample_hessian,
      compute_metrics_fn=compute_metrics,
      rng_key=subkey,
      initial_metrics=Metrics)

for epoch in range(10000):
  print(epoch)
  # ERROR? Shouldn't this be (p*p - test_set_size) / batch_size?
  state = training.train(batched(train_ds, batch_size), state, p*p / batch_size)
  training.test(batched(test_ds, batch_size), state)
