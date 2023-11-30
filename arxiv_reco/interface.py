import streamlit as st
from model_requests import call_translate_api, call_summary_api, call_recommend_api

# This file contains all the functions that app.py uses
# File that was uploaded to 06_app instance

def st_prompt():
    # Function that prompts user for keywords
    # input: user query
    # output: keywords
    st.write("ArXiv Recommendation System")
    keywords = st.text_input("Enter keywords to search papers on ArXiv")
    submit_button = st.button("Search")

    if submit_button and keywords:
        return keywords
    else:
        return None


def article_selectbox(articles, reco_list_key='reco_list'):
    # Fetching the current recommendation list from session state
    reco_list = st.session_state.get(reco_list_key, None)

    if reco_list is not None:
        options = [articles[idx]['title'] for idx in reco_list]
    else:
        options = [article['title'] for article in articles]

    selected_title = st.selectbox("Choose an article to inspect:", options, key='article_selectbox')

    # Find the index in articles that corresponds to the selected title
    if selected_title:
        if reco_list is not None:
            # Find the index within reco_list that corresponds to the selected title
            reco_idx = options.index(selected_title)
            # Map this index back to the original articles list
            selected_idx = reco_list[reco_idx]
        else:
            # If not using reco_list, find the index in the full articles list
            selected_idx = options.index(selected_title)

        st.session_state['selected_idx'] = selected_idx
    elif 'selected_idx' not in st.session_state:
        st.session_state['selected_idx'] = None

    if st.session_state['selected_idx'] is not None:
        selected_article = articles[st.session_state['selected_idx']]
        display_article_details(selected_article)

        if st.button("Translate", key='translate_' + str(st.session_state['selected_idx'])):
            translate_article(selected_article)

        if st.button("Summarise", key='summarize_' + str(st.session_state['selected_idx'])):
            summarize_article(selected_article)

        if st.button("Show Similar", key='reco_' + str(st.session_state['selected_idx'])):
            update_recommendations(st.session_state['graph'], st.session_state['selected_idx'], reco_list_key)

    else:
        st.write("No article has been chosen!")

    return st.session_state.get('selected_idx', None)


def display_article_details(article):
    st.write("Selected Article Details:")
    st.write(f"Title: {article['title']}")
    st.write(f"Summary: {article['summary']}")
    st.write(f"Link: {article['link']}")


def translate_article(article):
    st.write(f"Translated Title: {call_translate_api(article['title'])}")
    st.write(f"Translated Summary: {call_translate_api(article['summary'])}")


def summarize_article(article):
    st.write("AI-generated summary: ", call_summary_api(article['link']))


def update_recommendations(graph, selected_idx, reco_list_key):
    st.session_state[reco_list_key] = call_recommend_api(graph, selected_idx)
    # Reset the selected index as we now have a new list of articles
    st.session_state['selected_idx'] = None
    # Refresh the page to update the list
    st.rerun()


