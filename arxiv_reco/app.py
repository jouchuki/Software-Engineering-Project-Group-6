from arxiv_reco import prerequisites
from frontend import st_prompt_for_keywords, st_choose_article, st_article_details, st_r_inspection, st_selection_pipeline
import streamlit as st

# cd C:\Users\vsoko\PycharmProjects\SEProject\Software-Engineering-Project-Group-6\arxiv_reco
# streamlit run C:/Users/vsoko/PycharmProjects/SEProject/Software-Engineering-Project-Group-6/arxiv_reco/app.py


import streamlit as st

def main():
    # Step 1: Initialize Session State Variables
    if 'batch_size' not in st.session_state:
        st.session_state['batch_size'] = 10

    if 'keywords' not in st.session_state:
        st.session_state['keywords'] = None

    if 'articles' not in st.session_state:
        st.session_state['articles'] = None

    if 'reco_list' not in st.session_state:
        st.session_state['reco_list'] = None

    # Step 2: Keyword Input
    if not st.session_state['keywords']:
        st.session_state['keywords'] = st_prompt_for_keywords()

    # Step 3: Load Data and Prerequisites
    if st.session_state['keywords'] and not st.session_state['articles']:
        st.session_state['articles'], st.session_state['graph'], st.session_state['model'] = prerequisites(st.session_state['keywords'])

    # Step 4: Article Selection and Details Viewing
    if st.session_state['articles'] and not st.session_state['reco_list']:
        st.session_state['reco_list'] = st_selection_pipeline(st.session_state['articles'], st.session_state['batch_size'], st.session_state['graph'], st.session_state['model'])

    # Step 6: View Recommendations
    if st.session_state['reco_list']:
        st.write("Do you wish to inspect the most similar articles?")
        if st.button("Inspect Similar", key="inspect"):
            st_r_inspection(st.session_state['batch_size'], st.session_state['articles'], st.session_state['reco_list'])


if __name__ == "__main__":
    main()
