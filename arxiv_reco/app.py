from arxiv_reco import prerequisites
from frontend import st_prompt_for_keywords, st_choose_article, st_article_details, st_r_inspection, st_selection_pipeline
import streamlit as st

# cd C:\Users\vsoko\PycharmProjects\SEProject\Software-Engineering-Project-Group-6\arxiv_reco
# streamlit run C:/Users/vsoko/PycharmProjects/SEProject/Software-Engineering-Project-Group-6/arxiv_reco/app.py


def main():
    # Instantiate batch_size --- amount of articles shown in one batch
    if 'batch_size' not in st.session_state:
        st.session_state['batch_size'] = 10

    # Prompt for keywords
    if 'keywords' not in st.session_state:
        st.session_state['keywords'] = st_prompt_for_keywords()

        # Proceed only if keywords are provided
        if st.session_state.keywords:
            st.session_state['articles'], st.session_state['graph'], st.session_state['model'] = prerequisites(
                st.session_state['keywords'])

            # Call st_selection_pipeline and check the returned value
            if 'reco_list' not in st.session_state:
                st.session_state['reco_list'] = st_selection_pipeline(st.session_state.articles,
                                                                      st.session_state.batch_size,
                                                                      st.session_state.graph, st.session_state.model)

            # Ensure that result is not None
            if st.session_state.reco_list:
                st_r_inspection(st.session_state.reco_list, st.session_state.batch_size, st.session_state.graph)
            else:
                # Handle the case where result is None (e.g., show a message or take alternative action)
                st.write("No recommendations available for the given keywords.")

if __name__ == "__main__":
    main()
