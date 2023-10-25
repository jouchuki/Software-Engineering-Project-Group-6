import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv


def get_negative_samples(positive_samples, num_nodes, num_negative_samples):
    negative_samples = set()
    while len(negative_samples) < num_negative_samples:
        u, v = torch.randint(0, num_nodes, (2,)).tolist()
        if u != v and (u, v) not in positive_samples and (v, u) not in positive_samples:
            negative_samples.add((u, v))
    return list(negative_samples)

# Load the graph data
graph = torch.load("arxiv_graph.pt")

# Splitting the edges
edge_indices = graph.edge_index.t().cpu().numpy()
positive_samples = edge_indices.tolist()

def split_data(positive_samples, num_nodes, test_size=0.1, random_state=None):
    train_edges, test_edges, train_labels, test_labels = train_test_split(
        positive_samples, [1] * len(positive_samples), test_size=test_size, random_state=random_state
    )

    num_negative_samples_train = len(train_edges)
    negative_samples_train = get_negative_samples(positive_samples, num_nodes, num_negative_samples_train)

    num_negative_samples_test = len(test_edges)
    negative_samples_test = get_negative_samples(positive_samples, num_nodes, num_negative_samples_test)

    train_edges += negative_samples_train
    train_labels += [0] * num_negative_samples_train

    test_edges += negative_samples_test
    test_labels += [0] * num_negative_samples_test

    return torch.tensor(train_edges, dtype=torch.long).t(), torch.tensor(train_labels, dtype=torch.float), torch.tensor(test_edges, dtype=torch.long).t(), torch.tensor(test_labels, dtype=torch.float)

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

# Initialize model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModifiedEdgeClassifier(graph.num_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Data split
train_edges, train_labels, test_edges, test_labels = split_data(positive_samples, graph.x.size(0), random_state=123)
train_edges, train_labels = train_edges.to(device), train_labels.to(device)
test_edges, test_labels = test_edges.to(device), test_labels.to(device)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    predictions = torch.sigmoid(model(graph.x, train_edges).squeeze())
    loss = F.binary_cross_entropy(predictions, train_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
def test():
    model.eval()
    predictions = torch.sigmoid(model(graph.x, test_edges).squeeze())
    correct = ((predictions > 0.5) == test_labels).sum().item()
    return correct / len(test_labels)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    loss = train()
    acc = test()
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

# Save the model
torch.save(model.state_dict(), "arxiv_reco/ArxivReco.pth")