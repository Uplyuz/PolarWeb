
import plotly.express as px
import streamlit as st
import requests
import pandas as pd
from translate_API_output import traducir, traducir_tweets, preprocess_tweet
from datetime import datetime
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
from dashboard_charts import plot_wordcloud, sentiment_dist, format_data_model_output, obtain_summary, likes_over_words_amount, sentiment_dist_plotly, create_banner, format_number
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from api_handler import fetch_tweets_with_pagination, clean_entries_with_dates, extract_text_from_json_threads
import toml
import os

#--------------------------------------------------------------------------------------------------------------------------
import sys
sys.exit("Ejecución detenida para pruebas.")

# A partir de aquí, no se ejecutará nada
print("Este código no se ejecutará.")



#--------------------------------------------------------------------------------------------------------------------------

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

# Access secrets via Streamlit's st.secrets
api_key = st.secrets["x-rapidapi-key"]

# X API configuration
url_tweets_search_api_01 = "https://twitter-x.p.rapidapi.com/search/"
headers = {
    "x-rapidapi-key": api_key,  # Load API key from Streamlit secrets
    "x-rapidapi-host": "twitter-x.p.rapidapi.com"
}

# Threads API configuration ###########################

url_thread_latest = "https://threads-api4.p.rapidapi.com/api/search/recent"
url_thread_top = "https://threads-api4.p.rapidapi.com/api/search/top"

headers_threads = {
	"x-rapidapi-key": "2e2c904e0cmshdbad92d97808688p1e798ajsna8b04a4dd68a",
	"x-rapidapi-host": "threads-api4.p.rapidapi.com"
}

#########################


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

# to start session_state and state if the search is completed
if 'search_done' not in st.session_state:
    st.session_state.search_done = False
if 'df_clean_data' not in st.session_state:
    st.session_state.df_clean_data = None

# cover image
enlace_img="https://raw.githubusercontent.com/Uplyuz/PolarWeb/refs/heads/main/.streamlit/images/portrait.jpg"
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
        st.write('''
                
                ''')
        keyword = st.text_input("Enter a keyword to search tweets:", "ONU")
        st.write(' ')
        num_tweets = st.slider("Select the number of tweets to retrieve", 10, 100, step=10)

        st.markdown(
            """
            <div style="text-align: center; color: rgba(94, 255, 75, 0.8); font-size: 11px; font-family: monospace;">
            Please note that fetching more tweets may result in longer wait times.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(' ')
        option = st.radio('Tweet options', 
                    ('Latest', 'Top'), 
                        index=0, 
                        key='option', horizontal=True)
        st.write(' ')

        #  start searching 
        if st.button("Search"):
            if not keyword.strip():
                st.warning("You can't search with an empty keyword. Please enter a keyword")
            else:
                st.session_state.search_done = True  # Successful search
                user_search_phrase = keyword  # User input from the search box
                # API query string with a constant limit of 20
                querystring = {"query": user_search_phrase, "section": option.lower(), "limit": '50'}
                querystring_threads = {"query":user_search_phrase, "limit": str(num_tweets)}
                # calling APIs
                try:
                    # Use the pagination function to fetch tweets
                    max_tweets = num_tweets  # Set the limit based on user's slider input
                    tweets = fetch_tweets_with_pagination(url_tweets_search_api_01, querystring, headers, max_tweets)
                    
                    if option.lower() == 'top':
                        url_to_use = url_thread_top 
                    else:
                        url_to_use = url_thread_latest
                    response_threads = requests.get(url_to_use, headers=headers_threads, params=querystring_threads)
                    json_data_threads = response_threads.json()
                    threads = extract_text_from_json_threads(json_data_threads)
                    
                    # Convert fetched tweets-threads to DataFrame
                    df_clean_tweets = pd.DataFrame(tweets, columns=['Date', 'Tweet', 'Tweet_Likes']) 
                    df_clean_tweets['SocialN'] = 'X'
                    df_clean_threads = pd.DataFrame(threads, columns=['Date', 'Tweet', 'Tweet_Likes'])
                    df_clean_threads['SocialN'] = 'Threads'
                    df_clean_data = pd.concat([df_clean_tweets, df_clean_threads], ignore_index=True)
                    print(df_clean_data) ###########################################################################################

                    # Process and translate tweets
                    df_clean_data = preprocess_tweet(df_clean_data)
                    df_clean_data['Tweet'] = df_clean_data['Tweet'].apply(traducir)  # Translating tweet content
                    df_clean_data = df_clean_data.dropna(subset=['Tweet'])
                    df_clean_data = df_clean_data[df_clean_data['Tweet'].str.strip().astype(bool)]

                    # Store the cleaned data in session_state for further use
                    st.session_state.df_clean_data = df_clean_data

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
        st.markdown(
            """
            <div style="text-align: center; color: rgba(94, 255, 75, 0.8); font-size: 11px; font-family: monospace;">
            ⚠️ This app uses a roBERTa fine-tuned model, it may produce inaccurate results. ⚠️
            </div>
            """,
            unsafe_allow_html=True
        )

        # Applying the model in  Streamlit
        if st.session_state.search_done and model_custom is not None:
            df_clean_data = st.session_state.df_clean_data
            # Ensure DataFrame exists and has content before analysis
            if df_clean_data is not None and not df_clean_data.empty:
                # Analyze sentiments using the loaded model
                df_clean_data = analyze_sentiments(model_custom, tokenizer_custom, df_clean_data)
                
                # Update the session state with the new DataFrame
                st.session_state.df_clean_data = df_clean_data
                
        #  display results if search was successful
        if st.session_state.search_done:
            df_clean_data = st.session_state.df_clean_data
            # Check dataframe not none and empty
            if df_clean_data is not None and not df_clean_data.empty:
                if keyword.strip():
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