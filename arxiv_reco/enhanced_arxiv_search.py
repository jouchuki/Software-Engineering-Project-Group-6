from selection import prompt_for_keywords, selection_pipeline, r_inspection
from arxiv_reco import prerequisites

# Instantiate batch_size --- amount of articles shown in one batch
batch_size = 10


def main():
    while True:
        # Prompt for keywords
        keywords = prompt_for_keywords()

        articles, graph, model = prerequisites(keywords)

        # One function to control user interaction
        reco, reco_full = selection_pipeline(articles, batch_size, graph, model)

        # Function that lets user inspect recommendations
        r_inspection(reco_full, batch_size, graph)


if __name__ == "__main__":
    main()
