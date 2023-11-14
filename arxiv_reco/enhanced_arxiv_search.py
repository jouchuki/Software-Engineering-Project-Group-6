from selection import prompt_for_keywords, selection_pipeline, r_inspection
from arxiv_reco import prerequisites

# Instantiate batch size of amount of articles shown
batch_size = 10


def main():
    while True:
        # Prompt for keywords
        keywords = prompt_for_keywords()

        if keywords == 'exit':
            break

        articles, graph, model = prerequisites(keywords)

        # One function to control user interaction
        reco, reco_full = selection_pipeline(articles, batch_size, graph, model)

        # Function that lets user inspect recommendations
        r_inspection(reco_full, batch_size, graph)


if __name__ == "__main__":
    main()
