import streamlit as st
from pickle_util import read_temp_data, write_temp_data, delete_temp_file
import tempfile
from translate import trs_article
from summarisation import get_summary
from arxiv_reco import find_elements, recommend_for_article


class ArxivFE:
    def __init__(self, batch_size):
        self.batch_size = 10

    def st_prompt_for_keywords(self):
        st.write("ArXiv Recommendation System")

        keywords = st.text_input("Enter keywords to search papers on ArXiv", key="keywords_input")

        if st.button("Search") and keywords:
            return keywords


    def st_article_details(self, selected_idx, graph=None, articles=None, reco_mode=False):
        # Whether what data is used depends on what mode the function is run
        if reco_mode is False:
            # Paths to temporary files
            temp_art = tempfile.gettempdir() + '/articles_temp.pkl'
            temp_graph = tempfile.gettempdir() + '/graph_temp.pkl'
            # Read from pickle files if graph and articles are not provided
            if graph is None:
                graph = read_temp_data(temp_graph)

            if articles is None:
                articles = read_temp_data(temp_art)

            # Ensure that either graph.metadata or articles is available
            data_source = graph.metadata if graph and hasattr(graph, 'metadata') else articles
        else:
            temp_reco = tempfile.gettempdir() + '/reco_temp.pkl'
            data_source = read_temp_data(temp_reco)

        if data_source is not None and selected_idx < len(data_source):
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
            if st.button("Show Similar", key='reco_' + str(selected_idx)):
                return 0 # Placeholder for a show reco function

    def st_choose_article(self, batch_size, articles=None):
        temp_art = tempfile.gettempdir() + '/articles_temp.pkl'

        if articles is None:
            articles = read_temp_data(temp_art)

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
        if len(articles) > 10:
            if st.button('Next 10 Articles', key='next') and end_index < len(articles):
                st.session_state['current_article_index'] += batch_size
            elif st.button('Previous 10 Articles', key='back') and start_index > 0:
                st.session_state['current_article_index'] -= batch_size

        return selected_idx

    def st_selection_pipeline(self, articles, graph):
    # Checking if we need to refer to tempfiles
        if graph is None:
            temp_graph = tempfile.gettempdir() + '/graph_temp.pkl'
            graph = read_temp_data(temp_graph)

        if articles is None:
            temp_art = tempfile.gettempdir() + '/articles_temp.pkl'
            articles = read_temp_data(temp_art)

        selected_idx = self.st_choose_article(articles, self.batch_size)

        if selected_idx is not None:
            self.st_article_details(selected_idx, graph)
            reco = recommend_for_article(graph, selected_idx)
            reco_full = find_elements(articles, reco)
            temp_reco = tempfile.gettempdir() + '/reco_temp.pkl'
            write_temp_data(temp_reco, reco_full)
            selected_idx_reco = self.st_choose_article(self.batch_size, reco_full)
            if selected_idx_reco is not None:
                self.st_article_details(selected_idx_reco, reco_full, reco_mode=True)

        return None, None
