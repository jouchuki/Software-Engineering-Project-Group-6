# full directory command: cd C:\Users\vsoko\PycharmProjects\SEProject\Software-Engineering-Project-Group-6\arxiv_reco
# streamlit run C:/Users/vsoko/PycharmProjects/SEProject/Software-Engineering-Project-Group-6/arxiv_reco/app.py

import streamlit as st
from translate import trs_article
from summarisation import get_summary
from arxiv_reco import find_elements, recommend_for_article


class ArxivFrontend:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # Initialize session state variables if they don't exist
        if 'articles' not in st.session_state:
            st.session_state['articles'] = None
        if 'graph' not in st.session_state:
            st.session_state['graph'] = None
        if 'model' not in st.session_state:
            st.session_state['model'] = None
        if 'keywords' not in st.session_state:
            st.session_state['keywords'] = ''

    def st_prompt_for_keywords(self):
        st.write("ArXiv Recommendation System")
        keywords = st.text_input("Enter keywords to search papers on ArXiv", key="keywords_input")
        submit_button = st.button("Search")

        if submit_button and keywords:
            st.session_state['keywords'] = keywords
            return keywords
        elif 'keywords' in st.session_state:
            return st.session_state['keywords']
        return None

    def st_article_details(self, selected_idx, graph=None, articles=None):
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
                st.write("Summary: ", get_summary(article['link']))

            # Recommendation on demand

    def st_choose_article(self, articles, batch_size):
        if 'current_article_index' not in st.session_state:
            st.session_state['current_article_index'] = 0

        selected_idx = None
        start_index = st.session_state['current_article_index']
        end_index = min(start_index + batch_size, len(articles))

        # Display articles in the current batch with a "Go to Article" button
        for j in range(start_index, end_index):
            st.write(articles[j]['title'])
            if st.button('Check', key=f'go_{j}'):
                selected_idx = j
                return selected_idx

        # Navigation for batches
        if st.button('Next 10 Articles', key='next') and end_index < len(articles):
            st.session_state['current_article_index'] += batch_size
        elif st.button('Previous 10 Articles', key='back') and start_index > 0:
            st.session_state['current_article_index'] -= batch_size

        return selected_idx

    def st_r_inspection(self, reco_full, batch_size, graph):
        st.write("Now you can choose from recommended articles:")

        # This variable will keep track of the selected index for recommendations
        selected_idx_reco = None

        with st.expander("View Recommendations"):
            # Here we can call st_choose_article to let users browse recommendations
            selected_idx_reco = self.st_choose_article(reco_full, batch_size)
            # Save the selected index to the session state if needed
            if selected_idx_reco is not None:
                st.session_state['selected_reco_index'] = selected_idx_reco

        # Now check if there is a selected index from the recommendations
        if selected_idx_reco is not None:
            # Call st_article_details to show the details of the selected recommended article
            self.st_article_details(selected_idx_reco, articles=reco_full)
            # Additional buttons for interaction with the recommended article
            if st.button('Translate Article', key='translate_reco_' + str(selected_idx_reco)):
                trs_article(reco_full[selected_idx_reco]['link'])
            if st.button('Get AI-Generated Summary', key='summary_reco_' + str(selected_idx_reco)):
                get_summary(reco_full[selected_idx_reco]['link'])

    def st_selection_pipeline(self, articles, graph, model):
        selected_idx = self.st_choose_article(articles, self.batch_size)

        if selected_idx is not None:
            self.st_article_details(selected_idx, graph)
            reco = recommend_for_article(graph, model, selected_idx)
            reco_full = find_elements(articles, reco)
            return reco, reco_full
        return None, None
