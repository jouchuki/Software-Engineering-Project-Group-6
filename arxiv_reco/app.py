from interface import st_prompt, article_selectbox
import streamlit as st
from model_requests import call_get_data_api

# cd C:\Users\vsoko\PycharmProjects\SEProject\Software-Engineering-Project-Group-6\arxiv_reco
# streamlit run C:/Users/vsoko/PycharmProjects/SEProject/Software-Engineering-Project-Group-6/arxiv_reco/app.py


def main():
    # Step 1: Initialize Session State Variables
    if 'keywords' not in st.session_state:
        st.session_state['keywords'] = None

    if 'articles' not in st.session_state:
        st.session_state['articles'] = None

    if 'reco_list' not in st.session_state:
        st.session_state['reco_list'] = None

    # Step 2: Keyword Input
    if not st.session_state['keywords']:
        st.session_state['keywords'] = st_prompt()

    # Step 3: Load Data and Prerequisites
    if st.session_state['keywords'] is not None and not st.session_state['articles']:
        st.session_state['articles'], st.session_state['graph'] = call_get_data_api(st.session_state['keywords'])

    if st.session_state['articles']:
        article_selectbox(st.session_state['articles'])

    if st.button("Quit", key='quit'):
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    main()
