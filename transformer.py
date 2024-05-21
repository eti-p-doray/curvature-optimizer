import os
import requests
import numpy as np

import jax
import jax.numpy as jnp  # JAX NumPy

from flax import linen as nn  # Linen API
from flax import struct
import optax
from clu import metrics

import optimizers
import training

import time
import sys

# Obtain the learning rate from the command line arguments
scratch_dir = os.environ["SCRATCH"]
optimizer = sys.argv[1]
learning_rate = float(sys.argv[2])  # stepsize_

random_seed = 128
n_embed = 200 # Number of embedding dimensions
batch_size = 32 # How many independent sequences will we process in parallel?
block_size = 8 # What is the maximum context length for predictions?
num_heads = 4 # Number of heads in the multi-headed block

# Get the current time and use it to create a unique log filename
current_time = int(time.time())
log_directory = f'/home/mccrackn/curvature-optimizer/polynomials-mod-prime/conclusions/transformers/{optimizer}_n_embed={n_embed}'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)  # Create the directory if it doesn't exist  
log_filename = os.path.join(log_directory, f'{optimizer}_{learning_rate}_{current_time}.log')

# Open the log file and redirect stdout to it
sys.stdout = open(log_filename, 'w')



print(f"optimizer = {optimizer}")
print(f"learning_rate = {learning_rate}")
print(f"random_seed = {random_seed}")
print(f"n_embed = {n_embed}")
print(f"batch_size = {batch_size}")
print(f"block_size = {block_size}")
print(f"num_heads = {num_heads}")

# download the tiny shakespeare dataset
input_file_path = os.path.join(scratch_dir, 'data/shakespeare_char/input.txt')
os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
if not os.path.exists(input_file_path):
  data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
  with open(input_file_path, 'w') as f:
    f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
  data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
  return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
  l = np.array(l)
  return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = jnp.array(train_ids, dtype=jnp.uint16)
val_ids = jnp.array(val_ids, dtype=jnp.uint16)

def masked_fill(mask, a, fill):
  return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))

class FeedForward(nn.Module):
  """
  A feed forward multi-layer perceptron network.
  """
  n_embed: int

  @nn.compact
  def __call__(self, x):
    net = nn.Sequential([
      nn.Dense(4 * self.n_embed),
      jax.nn.relu,
      nn.Dense(self.n_embed)
    ])
    x = net(x)

    return x
    
class Head(nn.Module):
  """
  A single-headed self-attention decoder block.
  Takes the combined token and position embedding as input,
  then calculates the key and query values.
  The key and query are multiplied to calculate the 
  attention scores/affinities. The future weights are
  then altered to have zero affinity, this ensures the 
  model can't "cheat". The input is then used to calculate
  the values, which are then aggregated by multiplying 
  them with the weights.
  """
  head_size: int

  def setup(self):
    self.key = nn.Dense(self.head_size, use_bias=False)
    self.query = nn.Dense(self.head_size, use_bias=False)
    self.value = nn.Dense(self.head_size, use_bias=False)

  @nn.compact
  def __call__(self, x):
    B,T,C = x.shape
    
    k = self.key(x) # (B,T,C)
    q = self.query(x) # (B,T,C)
    # compute attention scores ("affinities")
    weights =  q @ k.transpose((0, -1, -2)) * self.head_size**-0.5 # (B, T, C) @ (B, C, T) ---> (B, T, T)
    tril = jnp.tril(jnp.ones(shape=(T, T), dtype=bool))
    tril = jnp.repeat(tril[None, ...], repeats=B, axis=0)
    weights = masked_fill(tril, weights, -jnp.inf)
    weights = jax.nn.softmax(weights, axis=-1)
    # perform the weighted aggregation of the values
    v = self.value(x)
    out = weights @ v
    return out
  
class MultiHeadedAttention(nn.Module):
  """
  Combines multiple heads of scaled self-attention 
  in parallel, then concatenates the heads outputs.
  """
  num_heads: int
  head_size: int
  n_embed: int

  @nn.compact
  def __call__(self, x):
    # Create a list of num_heads heads
    heads = [Head(self.head_size) for _ in range(self.num_heads)]
    # Provide the same input for each head
    heads_out = [h(x) for h in heads]
    combined_logits = jnp.concatenate(heads_out, axis=-1)
    # Perform a linear projection of the self-attention
    proj = nn.Dense(self.n_embed)
    logits = proj(combined_logits)
    return logits

class Block(nn.Module):
  """
  Transformer decoder block.
  It combines communication and computation.
  The communication is performed by the 
  multi-headed attention layer.
  Then the computation is performed by 
  the feed forward block.
  Skip connections are used to make the block scalable 
  and layer norm is used to speed up training.
  """
  n_embed: int
  num_heads: int

  @nn.compact
  def __call__(self, x):
    head_size = self.n_embed // self.num_heads
    sa_heads = MultiHeadedAttention(self.num_heads, head_size, self.n_embed)
    # Using skip connections with x + heads
    x = x + sa_heads(nn.LayerNorm()(x)) # apply one head of self-attention (B, T, C)
    ffwd = FeedForward(self.n_embed)
    x = x + ffwd(nn.LayerNorm()(x))
    return x
    
class AttentionLanguageModel(nn.Module):
  """
  Attention decoder language model.
  Uses the previous tokens in the sequence to 
  determine the probabilities of the next token.
  Processes the combined position and token embedding
  through multiple transformer decoder blocks, 
  which is then processed through a dense layer to 
  aquire the token logits.
  The logits can then be processed through a softmax
  function to calculate the token probabilities.
  """
  vocab_size: int
  n_embed: int
  block_size: int
  num_heads: int
  
  @nn.compact
  def __call__(self, index_seq):
    B, T = index_seq.shape

    # Each token directly reads off the logits for the next token from a lookup table
    token_embedding_table = nn.Embed(num_embeddings=self.vocab_size, features=self.n_embed) 
    token_emb = token_embedding_table(index_seq) # (B, T, C)

    position_embedding_table = nn.Embed(num_embeddings=self.block_size, features=self.n_embed) 
    pos_emb = position_embedding_table(jnp.arange(T)) # (T, C)

    x = token_emb + pos_emb # (B, T, C)

    blocks = nn.Sequential([
      Block(self.n_embed, num_heads=self.num_heads),
      Block(self.n_embed, num_heads=self.num_heads),
      Block(self.n_embed, num_heads=self.num_heads),
      nn.LayerNorm()
    ])
    x = blocks(x)

    lm_head = nn.Dense(self.vocab_size)
    logits = lm_head(x) # (B, T, vocab_size)

    return logits

  


rng_key = jax.random.PRNGKey(random_seed)

model = AttentionLanguageModel(vocab_size, n_embed, block_size, num_heads)
dummy_x = jnp.zeros(shape=(batch_size, block_size), dtype=jnp.uint16)
params = model.init(rng_key, dummy_x)['params']

def batched(data, rng_key, steps, batch_size, block_size):
  """
  Extracts a random batch of input and target data
  Args:
      data: An array of all the data's token ID's.
      rng_key: Random number generator key.
      batch_size: Number of parallel batches.
      block_size: Maximum time length for the token sequence.
  Returns:
      Input token ID's and target token ID's.
  """
  for step in range(steps):
    rng_key, subkey = jax.random.split(rng_key)
    ix = jax.random.randint(key=subkey, shape=(batch_size, ), minval=0, maxval=len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    yield x, y

def compute_metrics(metrics, *, loss, outputs, labels):
  metric_updates = metrics.single_from_model_output(
    logits=outputs[0], labels=labels, loss=loss)
  return metrics.merge(metric_updates)

@struct.dataclass
class Metrics(metrics.Collection):
  #accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')

def crossentrop_loss(y_pred, y):
  return optax.softmax_cross_entropy_with_integer_labels(logits=y_pred, labels=y).sum()

# Etienne was diviging LR by (batch_size * block_size)
if sys.argv[1] == "adam":
  tx = optax.adam(learning_rate=learning_rate)
if sys.argv[1] == "adamw":
  tx = optax.adamw(learning_rate=learning_rate, weight_decay = 0.0)
elif sys.argv[1] == "sgd":  
  tx = optax.sgd(0.5, 0.9)
elif sys.argv[1] == "kalman":
  tx = optimizers.kalman_blockwise_spectral_transformation(1.0, learning_rate, 16, 48, jax.random.PRNGKey(0))

opt_state = tx.init(params)

if sys.argv[1] == "kalman":
  state = training.NaturalTrainState(
    apply_fn=model.apply, params=params, tx=tx,
    opt_state=opt_state,
    loss_fn=crossentrop_loss,
    loss_hessian_fn=optimizers.sample_crossentropy_hessian,
    compute_metrics_fn=compute_metrics,
    rng_key=rng_key,
    initial_metrics=Metrics)
else:
  state = training.TrainState(
    apply_fn=model.apply, params=params, tx=tx,
    opt_state=opt_state,
    loss_fn=crossentrop_loss,
    loss_hessian_fn=optimizers.sample_crossentropy_hessian,
    compute_metrics_fn=compute_metrics,
    rng_key=rng_key,
    initial_metrics=Metrics)
  

rng_key, subkey = jax.random.split(rng_key)

# steps = 100
# for i in range(steps):
#   rng_key, subkey = jax.random.split(rng_key)
#   state = training.train(batched(train_ids, subkey, 144, batch_size, block_size), state, 144)
# rng_key, subkey = jax.random.split(rng_key)
# training.test(batched(val_ids, subkey, 144, batch_size, block_size), state)

for epoch in range(1000):
  print(f"epoch {epoch}")
  rng_key, subkey = jax.random.split(rng_key)
  state = training.train(batched(train_ids, subkey, 144, batch_size, block_size), state, 144)
  sys.stdout.flush()
rng_key, subkey = jax.random.split(rng_key)
training.test(batched(train_ids, subkey, 144, batch_size, block_size), state)
sys.stdout.flush()


# Close the redirected stdout file
sys.stdout.close()

# Restore stdout to its original setting if further printing to console is needed
sys.stdout = sys.__stdout__