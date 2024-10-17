import pandas as pd
import seaborn as sns
import streamlit as st  # Agregar esta línea
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np


# cloud charts (positive and negative sentiments) using Plotly
def plot_wordcloud(df, keyword):
    if 'Tweet' not in df.columns or 'Sentiment' not in df.columns:
        st.error("The Dataframe's structure is not correct.")
        return
    
    # Helper function to remove keyword from the tweets
    def remove_keyword(text, keyword):
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        return pattern.sub('', text)
    
    # Extract positive and negative tweets and remove the keyword
    positive_words = " ".join(df['Tweet'][df['Sentiment'] == 'Positive'])
    negative_words = " ".join(df['Tweet'][df['Sentiment'] == 'Negative'])
    
    positive_words = remove_keyword(positive_words, keyword)
    negative_words = remove_keyword(negative_words, keyword)

    # Generate word clouds for positive and negative words
    wordcloud_positive = WordCloud(width=800, height=400, background_color="rgba(255, 255, 255, 0)", mode="RGBA").generate(positive_words)
    wordcloud_negative = WordCloud(width=800, height=400, background_color="rgba(255, 255, 255, 0)", mode="RGBA").generate(negative_words)
    
    # Convert word clouds to images
    wordcloud_positive_image = wordcloud_positive.to_image()
    wordcloud_negative_image = wordcloud_negative.to_image()

    # Convert images to array
    positive_img_array = np.array(wordcloud_positive_image)
    negative_img_array = np.array(wordcloud_negative_image)

    # Create Plotly figures to display the word clouds
    fig = go.Figure()

    # Adding positive word cloud as an image
    fig.add_trace(go.Image(z=positive_img_array))
    fig.update_layout(
        title_text="Positive Tweets Word Cloud",
        title_x=0.5,
        margin=dict(l=0, r=0, t=50, b=0),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display the negative word cloud in a second figure
    fig2 = go.Figure()
    fig2.add_trace(go.Image(z=negative_img_array))
    fig2.update_layout(
        title_text="Negative Tweets Word Cloud",
        title_x=0.5,
        margin=dict(l=0, r=0, t=50, b=0),
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)
    

def sentiment_dist(df):
    if 'Tweet' not in df.columns or 'Sentiment' not in df.columns:
        st.error("the Dataframe's structure is not correct.")
        return
    
    sentiment_counts = df['Sentiment'].value_counts()
    plt.figure(figsize=(5, 3))  
    colors = ['green' if sentiment == 'Positive' else 'red' for sentiment in sentiment_counts.index]
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    st.pyplot(plt, use_container_width=True)  
    plt.clf()  
    

def likes_over_words_amount(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Words_count'], df['Tweet_Likes'], alpha=0.5, color='green')
    plt.title('Relationship Between Words in Tweets and Likes')
    plt.xlabel('Amount of words')
    plt.ylabel('Likes')
    st.pyplot(plt, use_container_width=True)  
    plt.clf() 

def format_data_model_output(df):
    df_clean_data = df.copy()
    if 'Unnamed: 0' in df_clean_data.columns:
        df_clean_data = df_clean_data.drop('Unnamed: 0', axis=1)
    df_clean_data['Date'] = pd.to_datetime(df_clean_data['Date'])
    df_clean_data['Year'] = df_clean_data['Date'].dt.year
    df_clean_data['Month'] = df_clean_data['Date'].dt.month
    df_clean_data['Week'] = df_clean_data['Date'].dt.isocalendar().week
    df_clean_data['Date'] = df_clean_data['Date'].dt.date
    df_clean_data['Words_count'] = df_clean_data['Tweet'].str.split().apply(len)
    return df_clean_data

def obtain_summary(df):
    frecuency_dates = df['Date'].value_counts()
    df_results = pd.DataFrame(frecuency_dates).reset_index()
    df_results.columns = ['Date', 'tweets_count']
    average_word_counts = []
    positive_ratio = []
    for date in df_results['Date']:
        # Filtrar df por fecha 
        filtered_tweets = df[df['Date'] == date]
        # suma y conteo de words_count
        total_words = filtered_tweets['Words_count'].sum()      # somo los totales
        count_tweets = filtered_tweets['Words_count'].count()   # cuento por fecha
        print(count_tweets)
        average = total_words / count_tweets if count_tweets > 0 else 0  
        average_word_counts.append(average)
        # Positives count
        total_positives = (filtered_tweets['Sentiment']=='Positive').sum()  
        print(total_positives)
        count_tweets = filtered_tweets['Sentiment'].count()  # total rows
        ratio = total_positives / count_tweets 
        positive_ratio.append(ratio)
    df_results['Average_word_count'] = average_word_counts
    df_results['Positive_ratio'] = positive_ratio
    summary = df_results.copy()
    return summary



def sentiment_dist_plotly(df):
    if 'Tweet' not in df.columns or 'Sentiment' not in df.columns:
        raise ValueError("The DataFrame's structure is not correct.")
    
    # Conteo de los valores de Sentiment
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    # Crear un gráfico de barras interactivo con Plotly
    fig = px.bar(sentiment_counts, 
                 x='Sentiment', 
                 y='Count', 
                 color='Sentiment', 
                 color_discrete_map={'Positive':'#2ECC71', 'Negative':'#E74C3C'},
                 title="Distribución de Sentimientos en Tweets",
                 labels={'Sentiment': 'Sentimiento', 'Count': 'Número de Tweets'},
                 text='Count',
                 height=500)

    # Actualizar el diseño del gráfico para una mejor presentación
    fig.update_traces(textposition='outside', marker_line_width=2, marker_line_color='black')
    fig.update_layout(
        title_font_size=24,
        title_x=0.5,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        font=dict(family="Arial", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
    )
    
    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)


