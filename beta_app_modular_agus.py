import plotly.express as px
import streamlit as st
import requests
import pandas as pd
from translate_API_output import traducir, traducir_tweets, preprocess_tweet
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from deep_translator import GoogleTranslator
import re
import streamlit as st
import torch
from transformers import RobertaTokenizer
from custom_class_final_model import CustomRobertaModel
from model_load_apply import load_custom_sentiment_model, predict_sentiment, analyze_sentiments
from dashboard_charts import plot_wordcloud, sentiment_dist, format_data_model_output, obtain_summary, likes_over_words_amount, sentiment_dist_plotly, create_banner
import plotly.graph_objects as go
from PIL import Image
import numpy as np

# CSS sidebar
st.markdown(
    """
    <style>
    /* Define siderbar background */
    [data-testid="stSidebar"] {
        background-color: #363737;  /* dark grey */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# API configuration
url_tweets_search_api_01 = "https://twitter-x.p.rapidapi.com/search/"
headers = {
    "x-rapidapi-key": "2e2c904e0cmshdbad92d97808688p1e798ajsna8b04a4dd68a", 
    "x-rapidapi-host": "twitter-x.p.rapidapi.com"
}

### MODEL  FROM HUGGINGfaces ###
@st.cache_resource
def get_model_and_tokenizer():
    try:
        model_custom, tokenizer_custom = load_custom_sentiment_model()
    except RuntimeError as e:
        st.error(str(e))
        return None, None
    
    return model_custom, tokenizer_custom

# Call the cached function
model_custom, tokenizer_custom = get_model_and_tokenizer()

# Check if the model was loaded successfully, otherwise exit
if model_custom is None or tokenizer_custom is None:
    st.stop()  # Stop the app if the model couldn't be loaded

# Function to clean the entries and extract date and text
def clean_entries_with_dates(list_of_elem):
    clean_data = []
    for element in list_of_elem:
        content = element.get('content',[])  # extract 'content'
        item_content = content.get('itemContent',{})  # extract'itemContent'
        if 'tweet_results' not in item_content or 'result' not in item_content['tweet_results'] : continue  # keep searching if not found:'tweet_result' or 'result'
        result = item_content['tweet_results']['result']  # extract 'tweet_results'
        if 'legacy' not in result or 'full_text' not in result['legacy']: continue  # keep searching if not found: 'full_text' o 'legacy'
        full_text = result['legacy']['full_text']  # extract 'full_text'
        post_date = result['legacy']['created_at']
        likes = result['legacy']['favorite_count']
        clean_data.append((post_date, full_text, likes))
    return clean_data

# to start session_state and state if the search is completed
if 'search_done' not in st.session_state:
    st.session_state.search_done = False
if 'df_clean_data' not in st.session_state:
    st.session_state.df_clean_data = None

# cover image
enlace_img = "https://raw.githubusercontent.com/Uplyuz/PolarWeb/refs/heads/main/.streamlit/images/portrait.jpg"
st.image(enlace_img, use_column_width=True)

# header
st.markdown("<p style='text-align: center; font-size:24px; font-monocode: bold;'>Tailored Sentiment Analysis at Your Fingertips</p>", unsafe_allow_html=True)

# create top-level tabs
first_level_tab = st.sidebar.radio("Select a tab", ["Data Analysis", "Heads Up"])

if first_level_tab == "Data Analysis":
    tab1, tab2 = st.tabs(["Set-up your Search", "Get Analysis"])

    # tab1: search config
    with tab1:
        st.markdown("<p style='text-align: center; font-size:20px; font-monocode: bold;'>Set up your search</p>", unsafe_allow_html=True)
        st.write(''' ''')
        keyword = st.text_input("Enter a keyword to search tweets:", "ONU")
        st.write(' ')
        num_tweets = st.slider("Select the number of tweets to retrieve", 20, 100, step=10)
        st.write(' ')
        option = st.radio('Tweet options', ('Latest', 'Top'), index=0, key='option', horizontal=True)
        st.write(' ')

        #  start searching
        if st.button("Search"):
            if not keyword.strip():
                st.warning("You can't search with an empty keyword. Please enter a keyword")
            else:
                # statement of actions to complete at the momento client click con button 'search'
                st.session_state.search_done = True  # successful search
                # to pass the keyword to the API as the search phrase
                user_search_phrase = keyword  # User input from the search box
                querystring = {"query": user_search_phrase, "section": option.lower(), "limit": '20'}  # Default filters (to be connected later with the st.slider and the st.radio)
                # calling the API
                try:
                    response_api_01 = requests.get(url_tweets_search_api_01, headers=headers, params=querystring)
                    response_api_01.raise_for_status()
                    entries_api_01 = response_api_01.json()['data']['search_by_raw_query']['search_timeline']['timeline']['instructions'][0]['entries']

                    # cleaning the API response data
                    clean_data = clean_entries_with_dates(entries_api_01)

                    # converting the cleaned data into a DataFrame
                    df_clean_data = pd.DataFrame(clean_data, columns=['Date', 'Tweet', 'Tweet_Likes'])
                    df_clean_data = preprocess_tweet(df_clean_data)
                    df_clean_data['Tweet'] = df_clean_data['Tweet'].apply(traducir)
                    df_clean_data = df_clean_data.dropna(subset=['Tweet'])
                    df_clean_data = df_clean_data[df_clean_data['Tweet'].str.strip().astype(bool)]
                    st.session_state.df_clean_data = df_clean_data

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                
        st.markdown(
            """
            <div style="color: rgba(94, 255, 75, 0.8); font-size: 11px; font-family: monospace;">
            ⚠️ This app uses a roBERTa fine tuned model, it may produce inaccurate results.
            </div>
            """,
            unsafe_allow_html=True
        )
            
        #  display results if search was successful
        if st.session_state.search_done:
            df_clean_data = st.session_state.df_clean_data
            # Check dataframe not none and empty
            if df_clean_data is not None and not df_clean_data.empty:
                st.write(' ')
                st.success("Task completed!")
                st.write(' ')
                st.subheader("Data Retrieved")
                st.write(" ")
                aux_01 = format_data_model_output(df_clean_data)
                st.write(aux_01)
            else:
                st.warning("No tweets were found for the current search.")

    # tab2: Analyzing data
    with tab2:
        st.subheader("Data Analysis")
    
        if st.session_state.search_done:
            df_clean_data = st.session_state.df_clean_data
            if df_clean_data is not None and not df_clean_data.empty:
                st.write("Sentiment Analysis Results:")
                aux_02 = obtain_summary(aux_01)
                create_banner(aux_01)
                sentiment_dist_plotly(df_clean_data)
                total_tweets = len(df_clean_data)
                total_likes = df_clean_data['Tweet_Likes'].sum()
                plot_wordcloud(df_clean_data, keyword)
                likes_over_words_amount(aux_01)
                st.write(f"Summary: ")
                st.write(aux_02)
            else:
                st.warning('Perform a search in tab "Set-up your Search" to get a personalized data analysis.')

# The second top-level tab 'Heads Up'
elif first_level_tab == "Heads Up":
    st.subheader("Heads Up")
    st.write("This is the second main tab.")
