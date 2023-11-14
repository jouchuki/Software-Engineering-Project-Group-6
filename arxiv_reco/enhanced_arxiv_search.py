import torch
from arxiv_reco import \
    query_arxiv, \
    construct_graph_from_embeddings, \
    ArxivReco, \
    recommend_for_article, \
    find_elements

from translate import ask_for_translation
from selection import article_details, choose_article, prompt_for_keywords, selection_pipeline
from summarisation import get_summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    while True:
        # Prompt for keywords
        keywords = prompt_for_keywords()

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

        # Function to scroll through articles
        selected_idx = choose_article(articles, batch_size)

        if selected_idx is not None:

            # Function to give user an option to select an article
            article_details(selected_idx, graph)

            # Ask the user if he would like to translate
            ask_for_translation(graph, selected_idx)

            # Ask the user if he would like to see AI-generated summary of the article
            get_summary(graph.metadata[selected_idx]['link'])

            # Get recommendations for the selected article
            reco = recommend_for_article(graph, model, selected_idx)
            reco = find_elements(articles, reco)

            # Chosen recommended articles' details
            print("Now you can choose from recommended articles:")
            selected_idx_reco = choose_article(reco, batch_size)
            article_details(selected_idx_reco, reco)


if __name__ == "__main__":
    main()