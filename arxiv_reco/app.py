from frontend_st import ArxivFE
from arxiv_reco import prerequisites
from pickle_util import read_temp_data, write_temp_data, delete_all_pkl_files
import tempfile
import streamlit as st
import sys

def main():
    sys.setrecursionlimit(50000)  # Be cautious with this
    delete_all_pkl_files('/')
    frontend = ArxivFE(batch_size=10)
    keywords = frontend.st_prompt_for_keywords()

    # File paths for articles and graph
    temp_art = tempfile.gettempdir() + '/articles_temp.pkl'
    temp_graph = tempfile.gettempdir() + '/graph_temp.pkl'

    if keywords:
        # Fetch new data
        articles, graph = prerequisites(keywords)
        # Save to temporary files
        write_temp_data(temp_art, articles)
        write_temp_data(temp_graph, graph)
    else:
        # Attempt to load from temporary files
        articles = read_temp_data(temp_art)
        graph = read_temp_data(temp_graph)

    # Proceed if we have articles and graph
    if articles and graph:
        result = frontend.st_selection_pipeline(articles, graph)

        if result:
            reco, reco_full = result
            # Handle recommendations display
            frontend.st_r_inspection(reco_full, frontend.batch_size, graph)

if __name__ == "__main__":
    main()

