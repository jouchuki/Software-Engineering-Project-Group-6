from arxiv_reco import prerequisites
from frontend import st_prompt_for_keywords, st_choose_article, st_article_details, st_r_inspection, st_selection_pipeline
import streamlit as st

# cd C:\Users\vsoko\PycharmProjects\SEProject\Software-Engineering-Project-Group-6\arxiv_reco
# streamlit run C:/Users/vsoko/PycharmProjects/SEProject/Software-Engineering-Project-Group-6/arxiv_reco/app.py


def main():
    # Instantiate batch_size --- amount of articles shown in one batch
    batch_size = 10

    # Prompt for keywords
    keywords = st_prompt_for_keywords()

    # Proceed only if keywords are provided and not 'exit'
    if keywords and keywords != 'exit':
        articles, graph, model = prerequisites(keywords)

        # Call st_selection_pipeline and check the returned value
        reco_list = st_selection_pipeline(articles, batch_size, graph, model)

        # Ensure that result is not None and unpack it
        if reco_list:
            st_r_inspection(reco_list, batch_size, graph)
        else:
            # Handle the case where result is None (e.g., show a message or take alternative action)
            st.write("No recommendations available for the given keywords.")

if __name__ == "__main__":
    main()
