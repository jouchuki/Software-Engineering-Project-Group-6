import torch
import json

# Load the PyG graph from the .pt file
loaded_graph = torch.load("arxiv_graph.pt")

# Print out the contents
print(loaded_graph)
