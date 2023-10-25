import torch
import requests
from bs4 import BeautifulSoup

ARXIV_API_URL = "http://export.arxiv.org/api/query?"

# 1. Fetch Data from ArXiv
def query_arxiv(keywords, max_results=10):
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
        articles.append({'title': title, 'summary': summary})

    return articles


# 2. Preprocess & Construct Graph
def construct_graph(articles):
    dataset = ArxivDataset(root='', raw_data=articles, preprocess=True)
    return dataset[0]  # Assuming the dataset returns a single graph


# 3. Predictions
def get_recommendations(graph, model, top_n=5):
    # Assuming the model has a forward method that takes in a graph and returns edge scores
    edge_scores = model(graph.x, graph.edge_index)

    # Getting the top_n edges (articles)
    _, top_edge_indices = edge_scores.topk(top_n)
    return [graph.edge_index[:, idx] for idx in top_edge_indices]  # Return source-target pairs for top edges


def main():
    # Load the model
    model_path = "arxiv_reco/ArxivReco.pth"
    model = ModifiedEdgeClassifier(graph.num_features).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    while True:
        print("\n--- ArXiv Recommendation System ---")
        keywords = input("Enter keywords to search on ArXiv (or 'exit' to quit): ")

        if keywords == 'exit':
            break

        articles = query_arxiv(keywords)
        graph = construct_graph(articles)

        # Display articles and let the user select one
        for idx, article in enumerate(articles):
            print(f"{idx}. {article['title']}")

        selected_idx = int(input("Select an article number for recommendations: "))

        # Here, you might want to modify the graph or use the selected_idx to generate specific recommendations
        # For simplicity, I'm using the entire graph
        recommendations = get_recommendations(graph, model)

        print("\nTop Recommendations:")
        for rec in recommendations:
            source, target = rec
            print(f"Source: {articles[source]['title']}, Target: {articles[target]['title']}")


if __name__ == "__main__":
    main()
