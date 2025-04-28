# Imports

# JAX Packages
import jax
import jax.numpy as jnp
import jax.scipy as jsp

import pandas as pd
import numpy as np
import optax
import itertools as it
import matplotlib.pyplot as plt
import json

"""# New Section"""

# Extract the node and edge data from our json files
try:
  with open("Data/converted_data_nodes.json", "r") as f:
    nodes = json.load(f)
except FileNotFoundError:
  print('Error opening the file: "converted_data_nodes.json"')
  exit(1)

try:
  with open("Data/converted_data_edges.json", "r") as f:
    edges = json.load(f)
except FileNotFoundError:
  print('Error opening the file: "converted_data_edges.json"')
  exit(1)

# Get the number of nodes(students) in the dataset
num_nodes = len(nodes)
print(num_nodes)

# Create an empty matrix to represent the data.
X_np = np.zeros((num_nodes,num_nodes),dtype=int)

# Fill the matrix with our edge data
for edge in edges:
  i = edge["source"]
  j = edge["target"]
  w = edge.get("weight", 1)
  X_np[i, j] += w

# strip array for a smaller sample size.
new = X_np[0:10, 0:10]

outgoing_weights = new.sum(axis=0)
incoming_weights = new.sum(axis=1)  # sum of each row
node = int(np.argmax(incoming_weights))

node2 = int(np.argmax(outgoing_weights))

max_outgoing = int(outgoing_weights[node])
max_weight = int(incoming_weights[node])

print(f"Node with the highest total incoming weight: Node {node} (Weight: {max_weight})")
print(f"Node with the highest total outgoing weight: Node {node2} (Weight: {max_outgoing})")

print(new)

# Generate a heatmap of our data
plt.imshow(new)
plt.colorbar()
plt.show()

# Convert to a JAX array
x = jnp.array(new)

def negative_log_likelihood(params, x: jnp.ndarray):
  """Calculate the log-likelihood of the Bradley Terry model."""
  n = x.shape[0]
  nll = 0
  for (i, j) in it.combinations(range(n), 2):
    nll += x[i, j] * jnp.log(jsp.special.expit(params[i] - params[j]))
    nll += x[j, i] * jnp.log(jsp.special.expit(params[j] - params[i]))
  return -nll

grad = jax.grad(negative_log_likelihood)

max_iterations = 20_000
output_every = 100
tolerance = 1e-4

solver = optax.adam(learning_rate=0.01)
params = jnp.array(np.random.normal(size=x.shape[0]))
opt_state = solver.init(params)

for t in range(max_iterations):
  grad_eval = grad(params, x)
  value = negative_log_likelihood(params, x)
  updates, opt_state = solver.update(grad_eval, opt_state)
  params = optax.apply_updates(params, updates)

  if t % output_every == 0:
    print(t, value, params, grad_eval)
    if (jnp.all(jnp.abs(grad_eval) < tolerance)):
      break

print("Scores λ:", params)
print("Strengths π:", np.exp(params))

strengths = np.exp(params)

# rank the strengths in descending order.
ranking = np.argsort(-strengths)

print(ranking)

print("Node Rankings from strongest to weakest:")
for rank, idx in enumerate(ranking):
    print(f"Rank {rank+1}: Node {idx} (Strength: {strengths[idx]:.3f})")
