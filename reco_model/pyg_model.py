import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

def get_negative_samples(positive_samples, num_nodes, num_negative_samples):
    negative_samples = set()
    while len(negative_samples) < num_negative_samples:
        u, v = torch.randint(0, num_nodes, (2,)).tolist()
        if u != v and (u, v) not in positive_samples and (v, u) not in positive_samples:
            negative_samples.add((u, v))
    return list(negative_samples)

# Load the graph data
graph = torch.load("arxiv_graph.pt")
#graph.x = torch.cat([graph.x_title, graph.x_summary], dim=-1)

# Splitting the edges
edge_indices = graph.edge_index.t().cpu().numpy()
positive_samples = edge_indices.tolist()
num_edges = len(positive_samples)

# Create train/test split
train_edges, test_edges, train_labels, test_labels = train_test_split(
    positive_samples, [1]*num_edges, test_size=0.1, random_state=42
)

# Get negative samples
num_negative_samples = len(train_edges)
negative_samples = get_negative_samples(positive_samples, graph.x.size(0), num_negative_samples)

# Append to training data
train_edges += negative_samples
train_labels += [0]*num_negative_samples

# Convert to tensors
train_edges = torch.tensor(train_edges, dtype=torch.long).t()
train_labels = torch.tensor(train_labels, dtype=torch.float)

print("Size of train_edges:", train_edges.size(1))
print("Size of train_labels:", len(train_labels))

test_edges = torch.tensor(test_edges, dtype=torch.long).t()
test_labels = torch.tensor(test_labels, dtype=torch.float)

# Define the GNN model
class ModifiedEdgeClassifier(torch.nn.Module):
    def __init__(self, num_features):
        super(ModifiedEdgeClassifier, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        self.classifier = torch.nn.Linear(2 * 32, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        start, end = edge_index
        edge_features = torch.cat([x[start], x[end]], dim=1)
        return self.classifier(edge_features)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModifiedEdgeClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def split_data(positive_samples, test_size=0.1, random_state=None):
    train_edges, test_edges, train_labels, test_labels = train_test_split(
        positive_samples, [1] * len(positive_samples), test_size=test_size, random_state=random_state
    )

    num_negative_samples = len(train_edges)
    negative_samples = get_negative_samples(positive_samples, graph.x.size(0), num_negative_samples)

    train_edges += negative_samples
    train_labels += [0] * num_negative_samples

    return torch.tensor(train_edges, dtype=torch.long).t(), torch.tensor(train_labels, dtype=torch.float), torch.tensor(
        test_edges, dtype=torch.long).t(), torch.tensor(test_labels, dtype=torch.float)

def recommend_articles_for_node(node_id, graph, model, top_n=5):
    model.eval()
    all_nodes = torch.arange(graph.x.size(0), dtype=torch.long)
    source_nodes = torch.full((graph.x.size(0),), node_id, dtype=torch.long)
    edge_candidates = torch.stack([source_nodes, all_nodes], dim=0)

    with torch.no_grad():
        edge_scores = torch.sigmoid(model(graph.x, edge_candidates).squeeze())

    edge_scores[node_id] = -float('inf')
    _, top_article_indices = edge_scores.topk(top_n)

    return top_article_indices.tolist()

def train(data):
    model.train()
    optimizer.zero_grad()
    predictions = torch.sigmoid(model(data.x, train_edges).squeeze())  # Note the change here
    loss = F.binary_cross_entropy(predictions, train_labels.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

def test(data):
    model.eval()
    predictions = torch.sigmoid(model(data.x, test_edges).squeeze())  # Note the change here
    correct = ((predictions > 0.5) == test_labels.to(device)).sum().item()
    return correct / len(test_labels)

train_edges, train_labels, test_edges, test_labels = split_data(positive_samples, random_state=123)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    loss = train(graph)
    acc = test(graph.to(device))
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

# Save the model
torch.save(model.state_dict(), "arxiv_reco/ArxivReco.pth")
