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

    if positive_words.strip():
        # Generate word clouds for positive and negative words
        wordcloud_positive = WordCloud(width=800, height=400, background_color="rgba(255, 255, 255, 0)", mode="RGBA").generate(positive_words)
        # Convert word clouds to images
        wordcloud_positive_image = wordcloud_positive.to_image()
        # Convert images to array
        positive_img_array = np.array(wordcloud_positive_image)

        # Create Plotly figures to display the word clouds
        fig = go.Figure()

        # Adding positive word cloud as an image
        fig.add_trace(go.Image(z=positive_img_array))
        fig.update_layout(
            title_text="Positive Posts Words Cloud",
            title_x=0,
            margin=dict(l=0, r=0, t=50, b=0),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No positive words were found.")

    if negative_words.strip():
        wordcloud_negative = WordCloud(width=800, height=400, background_color="rgba(255, 255, 255, 0)", mode="RGBA").generate(negative_words)
        # Convert word clouds to images
        wordcloud_negative_image = wordcloud_negative.to_image()
        # Convert images to array
        negative_img_array = np.array(wordcloud_negative_image)

        # Display the negative word cloud in a second figure
        fig2 = go.Figure()
        fig2.add_trace(go.Image(z=negative_img_array))
        fig2.update_layout(
            title_text="Negative Posts Words Cloud",
            title_x=0,
            margin=dict(l=0, r=0, t=50, b=0),
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No negative words were found.")

    

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
    # Create a scatter plot with Plotly
    fig = px.scatter(
        df,
        x='Words_count',  # X-axis will be the word count in the tweets
        y='Tweet_Likes',  # Y-axis will represent the number of likes
        size='Tweet_Likes',  # Size of the dots is proportional to the number of likes
        color='Tweet_Likes',  # Color is also based on the likes, for visual effect
        color_continuous_scale='Viridis',  # Using a professional continuous color scale
        title='Relationship Between Words in Tweets/Threads and Likes',
        labels={
            'Words_count': 'Amount of Words',
            'Tweet_Likes': 'Number of Likes'
        },
        hover_data={'Words_count': True, 'Tweet_Likes': True}  # Show additional information on hover
    )
    
    # Customizing layout to enhance readability
    fig.update_layout(
        xaxis_title='Number of Words in Posts',
        yaxis_title='Likes on Posts',
        template='plotly_white',  # Use a clean white template for a professional look
        height=600,  # Adjust height for better appearance
        margin=dict(l=40, r=40, t=40, b=40)
    )
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

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
    #para arreglar el error
    df_clean_data['Tweet_Likes'] = pd.to_numeric(df_clean_data['Tweet_Likes'], errors='coerce')
    df_clean_data['Tweet_Likes'] = df_clean_data['Tweet_Likes'].fillna(0)
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
        average = round(total_words / count_tweets, 2) if count_tweets > 0 else 0  
        average_word_counts.append(average)
        # Positives count
        total_positives = (filtered_tweets['Sentiment']=='Positive').sum()  
        count_tweets = filtered_tweets['Sentiment'].count()  # total rows
        ratio = round(total_positives / count_tweets , 2)
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
                 title="Distribution of Tweets/Threads Sentiment",
                 labels={'Sentiment': 'Sentiment', 'Count': 'Number of Posts'},
                 text='Count',
                 height=500)

    # Actualizar el diseño del gráfico para una mejor presentación
    fig.update_traces(textposition='outside', marker_line_width=2, marker_line_color='black')
    fig.update_layout(
        title_font_size=24,
        title_x=0,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        font=dict(family="Arial", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
    )
    
    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)

def format_number(num):
    if num >= 1_000_000:  # Si el número es un millón o más
        return "{:.2f}M".format(num / 1_000_000)  # Dividimos por 1 millón y formateamos
    elif num >= 1_000:  # Si el número es mil o más
        return "{:.2f}K".format(num / 1_000)  # Dividimos por 1 mil y formateamos
    else:
        return "{:.2f}".format(num)  # Para números menores a mil, mostrar normal

def create_banner(df):
        # Calculate the required metrics from the DataFrame
    total_tweets = df['Tweet'].count()
    total_likes = df['Tweet_Likes'].sum()
    total_likes_formatted = format_number(total_likes)
    avg_likes_per_tweet = total_likes / total_tweets if total_tweets > 0 else 0
    avg_likes_per_tweet_formatted = format_number(avg_likes_per_tweet)

    positive_sentiment = (df['Sentiment'] == 'Positive').mean() * 100
    avg_words_per_tweet = df['Words_count'].mean()
    avg_words_per_tweet_formatted = format_number(avg_words_per_tweet)

    # Create a layout with two rows
    st.write("<div style='text-align: center;'>", unsafe_allow_html=True)
    
    # First row: displaying four metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tweets", total_tweets)
    
    with col2:
        st.metric("Total Likes", total_likes_formatted)
    
    with col3:
        st.metric("Avg Likes per Tweet", avg_likes_per_tweet_formatted)
    
    with col4:
        st.metric("Avg Words per Tweet", avg_words_per_tweet_formatted)

    st.write("</div>", unsafe_allow_html=True)

    # Second row: displaying the gauge for Positive Sentiment %
    st.write("<div style='text-align: center;'>", unsafe_allow_html=True)
    
    # Create the Plotly gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=positive_sentiment,
        title={'text': "Positive Sentiment %"},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "black"},
            'bar': {'color': "green"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 100], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50}}))
    
    # Show the gauge centered
    st.plotly_chart(fig, use_container_width=True)

    st.write("</div>", unsafe_allow_html=True)




# Function to plot the trend of positive and negative tweets for each search over time
def plot_sentiment_trend_over_time(df1, df2, keyword1, keyword2):
    # Convert the 'Date' column to datetime (ensure it has both date and time)
    df1['Date'] = pd.to_datetime(df1['Date'])
    df2['Date'] = pd.to_datetime(df2['Date'])

    # Create a column for the date-hour combination (to group by both day and hour)
    df1['DateHour'] = df1['Date'].dt.floor('H')  # Floor to the nearest hour
    df2['DateHour'] = df2['Date'].dt.floor('H')

    # Group by the date-hour and sentiment for both datasets
    df1_grouped = df1.groupby(['DateHour', 'Sentiment']).size().unstack(fill_value=0).reset_index()
    df2_grouped = df2.groupby(['DateHour', 'Sentiment']).size().unstack(fill_value=0).reset_index()

    # Add columns for missing sentiment categories (in case not every hour has both positive and negative tweets)
    if 'Positive' not in df1_grouped:
        df1_grouped['Positive'] = 0
    if 'Negative' not in df1_grouped:
        df1_grouped['Negative'] = 0
    
    if 'Positive' not in df2_grouped:
        df2_grouped['Positive'] = 0
    if 'Negative' not in df2_grouped:
        df2_grouped['Negative'] = 0

    # Prepare the data for plotting
    df1_grouped['Keyword'] = keyword1  # Add keyword labels to distinguish between search 1 and 2
    df2_grouped['Keyword'] = keyword2

    # Rename the sentiment columns to show "Keyword: Positive/Negative"
    df1_grouped.rename(columns={'Positive': f'{keyword1}: Positive', 'Negative': f'{keyword1}: Negative'}, inplace=True)
    df2_grouped.rename(columns={'Positive': f'{keyword2}: Positive', 'Negative': f'{keyword2}: Negative'}, inplace=True)

    # Combine both datasets
    df_combined = pd.merge(df1_grouped[['DateHour', f'{keyword1}: Positive', f'{keyword1}: Negative']],
                           df2_grouped[['DateHour', f'{keyword2}: Positive', f'{keyword2}: Negative']],
                           on='DateHour', how='outer').fillna(0)  # Outer join to keep all hours, fill missing values with 0

    # Sort by DateHour to ensure the plot is ordered chronologically
    df_combined = df_combined.sort_values('DateHour')

    # Define colors for the lines
    colors = {
        f'{keyword1}: Positive': 'lightgreen',
        f'{keyword1}: Negative': 'lightcoral',
        f'{keyword2}: Positive': 'darkgreen',
        f'{keyword2}: Negative': 'darkred'
    }

    # Create the figure
    fig = go.Figure()

    # Add lines for each sentiment (positive/negative) for both searches
    for column in [f'{keyword1}: Positive', f'{keyword1}: Negative', f'{keyword2}: Positive', f'{keyword2}: Negative']:
        fig.add_trace(go.Scatter(
            x=df_combined['DateHour'],
            y=df_combined[column],
            mode='lines',
            name=column,
            line=dict(color=colors[column], width=2)
        ))

    # Update layout
    fig.update_layout(
        title=f"Trend of Positive and Negative Tweets Over Time ({keyword1} vs {keyword2})",
        xaxis_title="Time (Date and Hour)",
        yaxis_title="Number of Tweets",
        template='plotly_white',
        height=600,
        legend_title_text="Sentiment"
    )

    st.plotly_chart(fig, use_container_width=True)





# Population Pyramid Function

def population_pyramid(df1, df2, keyword1, keyword2):
    # Metrics for both tweets and likes
    metrics = ['Positive Tweets', 'Negative Tweets', 'Total Likes on extracted Tweets', 'Avg Likes per Tweet', 
               'Total Likes (Positive)', 'Total Likes (Negative)']
    
    # Data for both df1 and df2
    data1 = [
        (df1['Sentiment'] == 'Positive').sum(),
        (df1['Sentiment'] == 'Negative').sum(),
        df1['Tweet_Likes'].sum(),
        df1['Tweet_Likes'].mean(),
        df1[df1['Sentiment'] == 'Positive']['Tweet_Likes'].sum(),
        df1[df1['Sentiment'] == 'Negative']['Tweet_Likes'].sum()
    ]
    data2 = [
        (df2['Sentiment'] == 'Positive').sum(),
        (df2['Sentiment'] == 'Negative').sum(),
        df2['Tweet_Likes'].sum(),
        df2['Tweet_Likes'].mean(),
        df2[df2['Sentiment'] == 'Positive']['Tweet_Likes'].sum(),
        df2[df2['Sentiment'] == 'Negative']['Tweet_Likes'].sum()
    ]

    # Convert the data into a DataFrame for easier manipulation
    df_comparison = pd.DataFrame({
        'Metric': metrics,
        keyword1: data1,
        keyword2: data2
    })

    # Apply normalization (log scaling) for each row separately
    df_comparison[keyword1] = df_comparison[keyword1].apply(lambda x: np.log1p(x))  # log(1 + x)
    df_comparison[keyword2] = df_comparison[keyword2].apply(lambda x: np.log1p(x))  # log(1 + x)

    # Create the population pyramid
    fig = go.Figure()

    # Plot data for keyword1 on the left
    fig.add_trace(go.Bar(
        y=df_comparison['Metric'],
        x=df_comparison[keyword1] * -1,  # Left side for keyword1 (negative values)
        name=keyword1,
        orientation='h',
        marker_color='lightgreen',  # Use a tone of green
        text=df_comparison[keyword1].apply(lambda x: f"{np.expm1(x):,.0f}"),  # Reverse log scale for display, without decimal places
        textposition='inside'
    ))

    # Plot data for keyword2 on the right
    fig.add_trace(go.Bar(
        y=df_comparison['Metric'],
        x=df_comparison[keyword2],  # Right side for keyword2 (positive values)
        name=keyword2,
        orientation='h',
        marker_color='lightcoral',  # Use a tone of red
        text=df_comparison[keyword2].apply(lambda x: f"{np.expm1(x):,.0f}"),  # Reverse log scale for display, without decimal places
        textposition='inside'
    ))

    # Layout for the population pyramid
    fig.update_layout(
        title_text=f"Comparison of Positive/Negative Tweets and Likes - {keyword1} vs {keyword2}",
        barmode='overlay',
        template='plotly_white',
        xaxis_title="Log Scaled Values",
        yaxis_title="Metric",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)




