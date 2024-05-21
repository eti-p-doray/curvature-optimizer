import os
import sys
import time
from threading import Thread
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

import optax
from clu import metrics
from transformers import GPT2Tokenizer
from datasets import load_dataset

import optimizers
import training
import time 
from queue import Queue, Empty

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def load_and_prepare_dataset():
    cache_dir = "/home/mccrackn/scratch/huggingface/wikimedia/wikipedia"
    # Load the dataset with streaming enabled to avoid loading all data into memory
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", cache_dir=cache_dir, streaming=True)
    return dataset

def split_dataset(dataset, train_size=5767032, test_size=640782):
    # Directly use the hardcoded numbers to split the dataset
    print(f"Using hardcoded sizes:")
    print(f"Training set size: {train_size}")
    print(f"Test set size: {test_size}")

    # Split the dataset into training and testing sets using take and skip methods
    train_data = dataset['train'].take(train_size)
    test_data = dataset['train'].skip(train_size).take(test_size)

    return train_data, test_data

dataset = load_and_prepare_dataset()
train_data, test_data = split_dataset(dataset)

def tokenize_data(data_stream, output_queue, epoch_size, seed):
    count = 0
    # Set a consistent random seed to ensure reproducibility
    jax.random.PRNGKey(seed)
    for example in data_stream:
        if count >= epoch_size:
            break
        # Tokenize the example text
        tokenized_input = tokenizer(example['text'], return_tensors='np', padding='max_length', truncation=True, max_length=block_size)
        input_ids = tokenized_input['input_ids'].squeeze()  # Remove the extra dimension
        # Put the tokenized input IDs into the output queue
        output_queue.put(input_ids)
        count += 1

def batched(queue, batch_size, block_size):
    batch_data = []
    while True:
        try:
            tokenized_input = queue.get(timeout=10)
            batch_data.append(tokenized_input)
            if len(batch_data) == batch_size:
                x = jnp.stack(batch_data)
                y = jnp.roll(x, -1, axis=1)
                # Yield a batch of inputs and targets
                yield x, y
                batch_data = []
        except Empty:
            print("Queue is empty. Waiting for data...")
            continue

# Command-line arguments handling
if len(sys.argv) < 3:
    print("Usage: python script.py <optimizer> <learning_rate>")
    sys.exit(1)

optimizer = sys.argv[1]
learning_rate = float(sys.argv[2])
random_seed = 128
n_embed = 600  # Number of embedding dimensions
batch_size = 8  # Number of sequences processed in parallel
block_size = 16  # Maximum context length for predictions
num_heads = 8  # Number of attention heads
vocab_size = tokenizer.vocab_size  # Vocabulary size from tokenizer

data_queue_train = queue.Queue(maxsize=batch_size*2)
data_queue_test = queue.Queue(maxsize=batch_size*2)

# Calculate the number of steps based on batch_size
train_size = 5767032
test_size = 640782
train_steps = train_size // batch_size
test_steps = test_size // batch_size

# Setup logging
current_time = int(time.time())
log_directory = f'/home/mccrackn/curvature-optimizer/polynomials-mod-prime/conclusions/wikipedia/{optimizer}_n_embed={n_embed}'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
log_filename = os.path.join(log_directory, f'{optimizer}_{learning_rate}_{current_time}.log')
# sys.stdout = open(log_filename, 'w')
print(f"optimizer = {optimizer}")
print(f"learning_rate = {learning_rate}")
print(f"random_seed = {random_seed}")
print(f"n_embed = {n_embed}")
print(f"batch_size = {batch_size}")
print(f"block_size = {block_size}")
print(f"num_heads = {num_heads}")
print(f"Training steps per epoch: {train_steps}")
print(f"Testing steps per epoch: {test_steps}")

def masked_fill(mask, a, fill):
  return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))

class FeedForward(nn.Module):
    n_embed: int

    @nn.compact
    def __call__(self, x):
        x = nn.Sequential([
            nn.Dense(4 * self.n_embed),
            jax.nn.relu,
            nn.Dense(self.n_embed)
        ])(x)
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
    B, T, C = x.shape
    
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
    heads = [Head(self.head_size) for _ in range(self.num_heads)]
    heads_out = [h(x) for h in heads]
    combined_logits = jnp.concatenate(heads_out, axis=-1)
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
        x = x + sa_heads(nn.LayerNorm()(x))
        ffwd = FeedForward(n_embed=self.n_embed)
        x = x + ffwd(nn.LayerNorm()(x))
        return x
    
class AttentionLanguageModel(nn.Module):
    vocab_size: int
    n_embed: int
    block_size: int
    num_heads: int
    
    @nn.compact
    def __call__(self, index_seq):
        token_embedding_table = nn.Embed(num_embeddings=self.vocab_size, features=self.n_embed)
        token_emb = token_embedding_table(index_seq)

        position_embedding_table = nn.Embed(num_embeddings=self.block_size, features=self.n_embed)
        pos_emb = position_embedding_table(jnp.arange(self.block_size))
        
        x = token_emb + pos_emb
        
        blocks = nn.Sequential([
            Block(n_embed=self.n_embed, num_heads=self.num_heads),
            Block(n_embed=self.n_embed, num_heads=self.num_heads),
            Block(n_embed=self.n_embed, num_heads=self.num_heads),
            nn.LayerNorm()
        ])
        x = blocks(x)
        
        lm_head = nn.Dense(self.vocab_size)
        logits = lm_head(x)
        return logits

rng_key = jax.random.PRNGKey(random_seed)
model = AttentionLanguageModel(vocab_size, n_embed, block_size, num_heads)
dummy_x = jnp.zeros(shape=(batch_size, block_size), dtype=jnp.uint16)
params = model.init(rng_key, dummy_x)['params']

def compute_metrics(metrics, *, loss, outputs, labels):
  metric_updates = metrics.single_from_model_output(
    logits=outputs[0], labels=labels, loss=loss)
  return metrics.merge(metric_updates)

@struct.dataclass
class Metrics(metrics.Collection):
  loss: metrics.Average.from_output('loss')

def crossentrop_loss(y_pred, y):
  return optax.softmax_cross_entropy_with_integer_labels(logits=y_pred, labels=y).sum()

optimizers_map = {
    "adam": optax.adam(learning_rate=learning_rate),
    "adamw": optax.adamw(learning_rate=learning_rate, weight_decay=0.0),
    "sgd": optax.sgd(learning_rate=learning_rate, momentum=0.9),
    "kalman": optimizers.kalman_blockwise_spectral_transformation(1.0, learning_rate, 10, 10, jax.random.PRNGKey(0))
}
tx = optimizers_map.get(optimizer)
if tx is None:
    print(f"Unsupported optimizer {optimizer}. Exiting.")
    sys.exit(1)

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

start_time = time.time()

for epoch in range(1):  # Adjust according to your needs
    print(f"Epoch {epoch}")

    # Threads for data tokenization
    producer_thread_train = Thread(target=tokenize_data, args=(train_data, data_queue_train, train_size, random_seed))
    producer_thread_test = Thread(target=tokenize_data, args=(test_data, data_queue_test, test_size, random_seed + 1))
    producer_thread_train.start()
    producer_thread_test.start()

    test_gen = batched(data_queue_test, batch_size, block_size)
    state = training.train_with_eval(batched(data_queue_train, batch_size, block_size), state, train_steps, test_gen, 100, 10, start_time)
    sys.stdout.flush()

    producer_thread_train.join()
    producer_thread_test.join()

sys.stdout.close()
sys.stdout = sys.__stdout__