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
from api_handler import fetch_tweets_with_pagination, clean_entries_with_dates
import toml
import os

    
# Access secrets via Streamlit's st.secrets
api_key = st.secrets["x-rapidapi-key"]

# API configuration
url_tweets_search_api_01 = "https://twitter-x.p.rapidapi.com/search/"
headers = {
    "x-rapidapi-key": api_key,  # Load API key from Streamlit secrets
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

#  create all tabs 
tab1, tab2, tab3 = st.tabs(["Set-up your Search", "Get Data", "Get Analysis"])

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

    # continue_disabled = not keyword.strip() a ver si la quito
    
    #  start searching 
    if st.button("Search"):
        if not keyword.strip():
            st.warning("You can't search with an empty keyword. Please enter a keyword")
        else:
            st.session_state.search_done = True  # Successful search
            user_search_phrase = keyword  # User input from the search box

            # API query string with a constant limit of 20
            querystring = {"query": user_search_phrase, "section": option.lower(), "limit": '25'}
            
            try:
                # Use the pagination function to fetch tweets
                max_tweets = num_tweets  # Set the limit based on user's slider input
                tweets = fetch_tweets_with_pagination(url_tweets_search_api_01, querystring, headers, max_tweets)

                # Convert fetched tweets to DataFrame
                df_clean_data = pd.DataFrame(tweets, columns=['Date', 'Tweet', 'Tweet_Likes']) 

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
            
    #  display results if search was successful
    if st.session_state.search_done:
        df_clean_data=st.session_state.df_clean_data
        # Check dataframe not none and empty
        if df_clean_data is not None and not df_clean_data.empty:
            if keyword.strip():
                #st.write(f'Here you have a sample of your "{keyword}" tweets search')
                #st.write(df_clean_data.head(5))
                st.write(' ')
                st.success("Task completed!")
                st.write(' ')
                st.write("""Please proceed to the 'Get Data' tab to view the raw data, 
                           or to the 'Get Analysis' tab for a quick analysis of your results.""")
                
        else:
            st.warning("No tweets were found for the current search.")
        

# Applying the model in  Streamlit
if st.session_state.search_done and model_custom is not None:
    df_clean_data = st.session_state.df_clean_data

    # Ensure DataFrame exists and has content before analysis
    if df_clean_data is not None and not df_clean_data.empty:
        # Analyze sentiments using the loaded model
        df_clean_data = analyze_sentiments(model_custom, tokenizer_custom, df_clean_data)
        
        # Update the session state with the new DataFrame
        st.session_state.df_clean_data = df_clean_data
        
        # Refresh display after adding sentiment analysis
        #st.write("Sentiment Analysis Completed:")
        #st.write(st.session_state.df_clean_data.head())


# tab2: displaying the full dataset and giving the opportunity to download it, not much else
with tab2:
    st.subheader("Data Retrieved")
    if st.session_state.search_done:
        df_clean_data=st.session_state.df_clean_data
        if df_clean_data is not None and not df_clean_data.empty:
            st.write("If you need, here you can download the full data results...")
            aux_01 = format_data_model_output(df_clean_data) #chequear linea 208 y esta
            st.write(aux_01)
            st.write(" ")
            st.write("If you are looking for a comprehensive data analysis of this results, please go to the 'Get Analysis' tab placed on header.")
    else:
        st.warning("No data available to display")
        pass




# tab3: Analysing data
with tab3:
    st.subheader("Data Analysis")
    
    if st.session_state.search_done:
        df_clean_data = st.session_state.df_clean_data
        if df_clean_data is not None and not df_clean_data.empty:
            
            # Apply sentiment analysis and add the sentiment column
            df_clean_data = analyze_sentiments(model_custom, tokenizer_custom, df_clean_data)
            aux_01 = format_data_model_output(df_clean_data)  # Format data for dashboard
            aux_02 = obtain_summary(aux_01)  # Obtain summary for reporting

            # Perform analysis and visualization
            create_banner(aux_01)  # Display key stats
            sentiment_dist_plotly(df_clean_data)  # Sentiment distribution plot
            plot_wordcloud(df_clean_data, keyword)  # Word clouds for positive and negative sentiments
            likes_over_words_amount(aux_01)  # Scatter plot of likes vs. word count

            st.write(f"Summary:")
            st.write(aux_02)
        else:
            st.warning("No data available for analysis.")
    else:
        st.warning('Perform a search in the "Set-up your Search" tab to get a personalized data analysis.')
