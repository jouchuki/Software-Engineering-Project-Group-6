#full directory command: cd C:\Users\vsoko\PycharmProjects\SEProject\Software-Engineering-Project-Group-6\arxiv_reco
#streamlit run C:/Users/vsoko/PycharmProjects/SEProject/Software-Engineering-Project-Group-6/arxiv_reco/app.py

import streamlit as st
from translate import trs_article
from summarisation import get_summary
from arxiv_reco import find_elements, recommend_for_article


def st_prompt_for_keywords():
    st.write("ArXiv Recommendation System")
    keywords = st.text_input("Enter keywords to search papers on ArXiv")
    submit_button = st.button("Search")

    if submit_button and keywords:
        return keywords
    else:
        return None


def st_article_details(selected_idx, graph=None, articles=None):
    # Determine the source of the data
    data_source = graph.metadata if graph is not None else articles

    if data_source is not None:
        article = data_source[selected_idx]

        # Display article details
        st.write("Selected Article Details:")
        st.write(f"Title: {article['title']}")
        st.write(f"Summary: {article['summary']}")
        st.write(f"Link: {article['link']}")

        # Button for translation on demand
        if st.button("Translate", key='translate_' + str(selected_idx)):
            st.write(f"Translated Title: {trs_article(article['title'])}")
            st.write(f"Translated Summary: {trs_article(article['summary'])}")

        # Button for summarization on demand
        if st.button("Summarise", key='summarize_' + str(selected_idx)):
            get_summary(article['link'])


def st_choose_article(articles, batch_size):
    selected_idx = None
    start_index = 0

    while True:
        end_index = min(start_index + batch_size, len(articles))

        # Display articles in the current batch with a "Go to Article" button
        for j in range(start_index, end_index):
            st.write(articles[j]['title'])
            if st.button('Check', key=f'go_{j}'):
                selected_idx = j
                return selected_idx

        # Navigation for batches
        if start_index + batch_size < len(articles) and st.button('Next 10 Articles', key='next'):
            start_index += batch_size
        elif start_index > 0 and st.button('Previous 10 Articles', key='back'):
            start_index -= batch_size
        else:
            break
    return selected_idx


def st_r_inspection(reco_full, batch_size, graph):
    st.write("Now you can choose from recommended articles:")

    # Using a refactored Streamlit version of choose_article
    selected_idx_reco = st_choose_article(reco_full, batch_size)

    if selected_idx_reco is not None:
        # Display details of the selected article
        st_article_details(selected_idx_reco, articles=reco_full)

        # Ask for translation and summary
        if st.button('Translate Article', key='translate_' + str(selected_idx_reco)):
            trs_article(graph.metadata[selected_idx_reco]['link'])

        if st.button('Get AI-Generated Summary', key='summary_' + str(selected_idx_reco)):
            get_summary(graph.metadata[selected_idx_reco]['link'])


def st_selection_pipeline(articles, batch_size, graph, model):
    selected_idx = st_choose_article(articles, batch_size)

    if selected_idx is not None:
        # Function to give user an option to select an article
        st_article_details(selected_idx, graph)

        # Get recommendations for the selected article
        reco = recommend_for_article(graph, model, selected_idx)
        reco_full = find_elements(articles, reco)
        return reco, reco_full
