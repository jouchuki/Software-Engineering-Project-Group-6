import json
import torch
import random
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def visualize(graph, label_to_category):
    G = nx.Graph()
    for edge in graph.edge_index.t().tolist():
        G.add_edge(edge[0], edge[1])
    color_map = [label_to_category[cat] for cat in graph.y.tolist()]
    plt.figure(figsize=(12, 12))
    nx.draw(G, node_color=color_map, with_labels=False, node_size=20, cmap=plt.cm.jet)
    plt.show()

def get_negative_samples(positive_samples, num_nodes, num_negative_samples):
    negative_samples = set()
    while len(negative_samples) < num_negative_samples:
        u, v = torch.randint(0, num_nodes, (2,)).tolist()
        if u != v and (u, v) not in positive_samples and (v, u) not in positive_samples:
            negative_samples.add((u, v))
    return list(negative_samples)

# Load preprocessed data
with open('arxiv_data_preprocessed_roberta.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

node_features_title = []
node_features_summary = []
node_labels = []
category_to_label = {category: i for i, category in enumerate(data.keys())}
label_to_category = {i: category for category, i in category_to_label.items()}

# Populate the node features and labels
for category, articles in data.items():
    for article in articles:
        node_features_title.append(article['title_embedding'])
        node_features_summary.append(article['summary_embedding'])
        node_labels.append(category_to_label[category])

# Convert lists to tensors
x_title = torch.tensor(node_features_title, dtype=torch.float)
x_summary = torch.tensor(node_features_summary, dtype=torch.float)
x = torch.cat((x_title, x_summary), dim=1)
y = torch.tensor(node_labels, dtype=torch.long)

# Obtain edges based on cosine similarity threshold
THRESHOLD = 0.9
edges = set()
for category, articles in data.items():
    embeddings = [a["title_embedding"] for a in articles]
    if not embeddings:
        continue
    similarities = cosine_similarity(embeddings)
    for i in range(len(articles)):
        for j in range(i+1, len(articles)):
            if similarities[i][j] >= THRESHOLD:
                edges.add((i, j))
                edges.add((j, i))
edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

# Create a dataset for edge prediction
positive_samples = list(edges)
negative_samples = get_negative_samples(positive_samples, x.size(0), len(positive_samples))
train_data = positive_samples + negative_samples
labels = [1] * len(positive_samples) + [0] * len(negative_samples)

# Shuffle data
combined = list(zip(train_data, labels))
random.shuffle(combined)
train_data, labels = zip(*combined)
train_data = torch.tensor(train_data, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.float)

# Create an initial PyG graph
graph = Data(x=x, edge_index=edge_index, y=y)

# Visualize the graph
#visualize(graph, label_to_category)

# Save the graph as a .pt file
torch.save(graph, "arxiv_graph.pt")
