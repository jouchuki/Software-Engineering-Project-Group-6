import torch
import requests
from bs4 import BeautifulSoup
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import RobertaTokenizer, RobertaModel
from torch_geometric.data import Data
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

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

def get_roberta_embedding(text, model=roberta_model, tokenizer=tokenizer):
    # Tokenize the text and convert it to tensor format
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract [CLS] token's embedding and return
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist()


ARXIV_API_URL = "http://export.arxiv.org/api/query?"

# 1. Fetch Data from ArXiv
def query_arxiv(keywords, max_results=100):
    query = f"search_query=all:{keywords}&start=0&max_results={max_results}"
    response = requests.get(ARXIV_API_URL + query)

    if response.status_code != 200:
        raise Exception("Failed to fetch data from ArXiv")

    # Parsing the XML using BeautifulSoup
    soup = BeautifulSoup(response.text, 'xml')
    entries = soup.find_all('entry')

    articles = []
    for entry in entries:
        title = entry.title.string
        summary = entry.summary.string
        link = entry.id.string
        articles.append({
        'title': title,
        'summary': summary,
        'link': link,
        'title_embedding': get_roberta_embedding(title),
        'summary_embedding': get_roberta_embedding(summary)
    })

    return articles

# 2. Preprocess & Construct Graph
def construct_graph_from_embeddings(articles):
    node_features_title = [article['title_embedding'] for article in articles]
    node_features_summary = [article['summary_embedding'] for article in articles]

    x_title = torch.tensor(node_features_title, dtype=torch.float)
    x_summary = torch.tensor(node_features_summary, dtype=torch.float)
    x = torch.cat((x_title, x_summary), dim=1)

    # Storing metadata as a list of dictionaries for each node (article)
    metadata = [{'title': article['title'], 'summary': article['summary'], 'link': article['link']} for article in articles]

    graph = Data(x=x, metadata=metadata)  # Attach the metadata to the graph
    return graph

# 3. Predictions
def recommend_for_article(graph, model, article_index, num_recommendations=10):
    # For all possible edges between the given article and all other articles
    all_other_articles = torch.arange(graph.x.size(0), device=device)
    all_edges = torch.stack([torch.full_like(all_other_articles, article_index), all_other_articles])

    # Predict the likelihood of relationships
    with torch.no_grad():
        predictions = torch.sigmoid(model(graph.x, all_edges).squeeze())

    # Filter out the input article index from the recommendations
    mask = all_other_articles != article_index
    filtered_predictions = predictions[mask]
    filtered_indices = all_other_articles[mask]

    # Sort by prediction score
    _, sorted_relative_indices = filtered_predictions.sort(descending=True)
    sorted_indices = filtered_indices[sorted_relative_indices]

    # Return the top article indices
    return sorted_indices[:num_recommendations].cpu().numpy()

def main():
    while True:
        print("\n--- ArXiv Recommendation System ---")
        keywords = input("Enter keywords to search on ArXiv (or 'exit' to quit): ")

        if keywords.isdigit():
            raise Exception('Numbers are not keywords!')

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
