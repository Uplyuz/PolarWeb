import pandas as pd
import seaborn as sns
import streamlit as st  # Agregar esta línea
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# cloud charts (positive and negative sentiments)
def plot_wordcloud(df):
    if 'Tweet' not in df.columns or 'Sentiment' not in df.columns:
        st.error("the Dataframe's structure is not correct.")
        return
    
    positive_words = " ".join(df['Tweet'][df['Sentiment'] == 'Positive'])
    negative_words = " ".join(df['Tweet'][df['Sentiment'] == 'Negative'])

    wordcloud = WordCloud(width=875, height=900, background_color="lightgrey", max_words=50, min_font_size=20, random_state=42)\
        .generate(positive_words)
    
    wordcloud2 = WordCloud(width=875, height=900, background_color="black", max_words=50, min_font_size=20, random_state=42)\
        .generate(negative_words)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), facecolor=None)  # Changed to 2 rows, 1 column
    ax1.imshow(wordcloud, interpolation='bilinear')
    ax2.imshow(wordcloud2, interpolation='bilinear')
    ax1.set_title('Positive Tweets', fontsize=20)
    ax2.set_title('Negative Tweets', fontsize=20)
    ax1.axis("off")
    ax2.axis("off")
    fig.tight_layout()
    
    st.pyplot(fig)  # don't take this out
    

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