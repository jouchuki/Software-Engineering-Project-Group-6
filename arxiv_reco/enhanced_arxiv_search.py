import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import warnings
from arxiv_reco import get_roberta_embedding, query_arxiv, construct_graph_from_embeddings, recommend_for_article

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.simplefilter(action='ignore', category=UserWarning)

class ArxivReco(torch.nn.Module):
    def __init__(self, num_features):
        super(ArxivReco, self).__init__()
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

def main():
    while True:
        print("\n--- ArXiv Recommendation System ---")
        keywords = input("Enter keywords to search on ArXiv (or 'exit' to quit): ")

        if keywords.isdigit():
            raise Exception('Numbers are not keywords!')
            break

        if keywords == 'exit':
            break

        articles = query_arxiv(keywords)
        graph = construct_graph_from_embeddings(articles)
        model_path = "ArxivReco.pth"
        model = ArxivReco(graph.x.size(1)).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Display articles and let the user select one
        batch_size = 10
        selected_idx = None

        for i in range(0, len(articles), batch_size):
            for j in range(i, min(i + batch_size, len(articles))):
                print(f"{j}. {articles[j]['title']}")

            # Check if it's the last batch; if so, only provide the article selection prompt
            if i + batch_size >= len(articles):
                user_input = input(
                    f"\nSelect an article number from ({len(batch_size) - 1}) for recommendations, or type 'exit' to search again: ")
            else:
                user_input = input(
                    f"\nSelect an article number from ({len(articles) - 1}) for recommendations, type 'n' for the next 10 articles, or 'exit' to search again: ")

            if user_input == 'n':
                continue
            elif user_input == 'exit':
                break
            elif user_input.isdigit() and 0 <= int(user_input) < len(articles):
                selected_idx = int(user_input)
                break

        if selected_idx is not None:
            # Display the details of the selected article
            print("\nSelected Article Details:")
            print(f"Title: {graph.metadata[selected_idx]['title']}")
            print(f"Summary: {graph.metadata[selected_idx]['summary']}")
            print(f"Link: {graph.metadata[selected_idx]['link']}")

            # Get recommendations for the selected article
            recommended_indices = recommend_for_article(graph, model, selected_idx, num_recommendations=10)

            print("\nTop Recommendations:")
            for idx in recommended_indices:
                print(graph.metadata[idx]['title'])

if __name__ == "__main__":
    main()
