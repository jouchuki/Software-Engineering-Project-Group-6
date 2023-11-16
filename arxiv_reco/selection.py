from translate import ask_for_translation
from summarisation import get_summary
from arxiv_reco import recommend_for_article, find_elements
import streamlit as st

def prompt_for_keywords():
    print("\n--- ArXiv Recommendation System ---")
    keywords = input("Enter keywords to search on ArXiv (or 'exit' to quit): ")

    if keywords.isdigit():
        raise Exception('Numbers are not keywords!')

    else:
        return keywords


def article_details(selected_idx, graph=None,  articles=None):
    # Display the details of the selected article
    if articles is None and graph is not None:
        print("\nSelected Article Details:")
        print(f"Title: {graph.metadata[selected_idx]['title']}")
        print(f"Summary: {graph.metadata[selected_idx]['summary']}")
        print(f"Link: {graph.metadata[selected_idx]['link']}")
    if graph is None and articles is not None:
        print("\nSelected Article Details:")
        print(f"Title: {articles[selected_idx]['title']}")
        print(f"Summary: {articles[selected_idx]['summary']}")
        print(f"Link: {articles[selected_idx]['link']}")



def choose_article(articles, batch_size):
    selected_idx = None
    for i in range(0, len(articles), batch_size):
        for j in range(i, min(i + batch_size, len(articles))):
            print(f"{j}. {articles[j]['title']}")

        if i + batch_size >= len(articles):
            user_input = input(f"\nSelect an article number from 0 to {len(articles) - 1}, or type 'exit' to search again: ")
        else:
            user_input = input(f"\nSelect an article number from 0 to {len(articles) - 1}, type 'n' for the next 10 articles, or 'exit' to search again: ")

        if user_input == 'n':
            continue
        elif user_input == 'exit':
            break
        elif user_input.isdigit() and 0 <= int(user_input) < len(articles):
            selected_idx = int(user_input)
            break
        else:
            print("Invalid input. Please try again.")

    return selected_idx


def selection_pipeline(articles, batch_size, graph, model):
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
        reco_full = find_elements(articles, reco)
        answer = input("Do you wish to inspect the recommended articles?(y/n)")
        if answer == "y":
            return reco, reco_full

        else:
            return 0

def r_inspection(reco_full, batch_size, graph):
    # Chosen recommended articles' details
    print("Now you can choose from recommended articles:")
    selected_idx_reco = choose_article(reco_full, batch_size)
    article_details(selected_idx_reco, articles=reco_full)

    # Ask the user if he would like to translate
    ask_for_translation(graph, selected_idx_reco)

    # Ask the user if he would like to see AI-generated summary of the article
    get_summary(graph.metadata[selected_idx_reco]['link'])