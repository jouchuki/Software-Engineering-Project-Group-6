from translate import ask_for_translation
from arxiv_reco import recommendations


def prompt_for_keywords():
    print("\n--- ArXiv Recommendation System ---")
    keywords = input("Enter keywords to search on ArXiv (or 'exit' to quit): ")

    if keywords.isdigit():
        raise Exception('Numbers are not keywords!')

    else:
        return keywords


def article_details(selected_idx, graph):
    # Display the details of the selected article
    print("\nSelected Article Details:")
    print(f"Title: {graph.metadata[selected_idx]['title']}")
    print(f"Summary: {graph.metadata[selected_idx]['summary']}")
    print(f"Link: {graph.metadata[selected_idx]['link']}")


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


def selection_pipeline(selected_idx, graph, model, batch_size):
    # Function to give user an option to select an article
    article_details(selected_idx, graph)

    # Ask the user if he would like to translate
    ask_for_translation(graph, selected_idx)

    # Get recommendations for the selected article
    print("Searching for recommended articles...\n")
    reco = recommendations(graph, model, selected_idx)

    # Chosen recommended articles' details
    selected_idx_reco = choose_article(reco, batch_size)
    article_details(selected_idx_reco, reco)
    ask_for_translation(reco, selected_idx_reco)