 ###4 Geeks Accademy, Final Project###

 Agustin, Luis, Alessandro. 

App URL:   https://polarweb-huds7zhggkgsrmrgg2pnxu.streamlit.app/

##### POLAR WEB #####
is an app that allows users to search for tweets on Twitter and perform sentiment analysis using a custom fine-tuned RoBERTa model. 
The app retrieves tweets through the Twitter API based on user defined inputs, applies sentiment analysis, and provides data visualization features as word clouds and bar charts to display the sentiment breakdown.

###Features overview###
Search Tweets: users can search for tweets using a keyword or phrase.
Customizable Search Options: Users can filter tweets by "Top" or "Latest" 
Sentiment Analysis: A fine-tuned RoBERTa model (detailed in the model section) is developed to classify the sentiment of each tweet as either Positive or Negative.
Data Visualization: The app includes a dashboard for deeper insights into the tweets.
Downloadable Dataset: Users can download the retrieved and analyzed tweet data as a CSV format file.
Language Translation: Tweets are translated to english, as the original and fine-tuned model were trained ins this language.
Heads up: A comparison between two searches.

####App layout####

1. Set-up your Search
In this tab, users configure their search options for retrieving tweets:

Keyword Input: Enter the keyword (more than 3 characters) or phrase you want to search for in tweets.
Tweet Type: Choose between "Top" tweets or "Latest" tweets using radio buttons.
Once the search is configured, the app retrieves tweets using X API and processes the response. 
Users are informed of the number of tweets retrieved and prompted to move to the next tab for data analysis.

Also in this tab, the retrieved tweets will be displayed in a table format, with the following columns:

Date: The timestamp when the tweet was posted.
Tweet: The full content of the tweet (translated).
Tweet_Likes: The number of likes the tweet received.
SocialN: The social network in which the tweet was posted.
Sentiment: The result of sentiment analysis (0 for Negative, 1 for Positive).
Year: The year when the tweet was posted.
Month: The month when the tweet was posted.
Week: The week when the tweet was posted.
Words_count: The number of word in the tweet.

Users can download the dataset with sentiment analysis as a CSV file.

2. Get Analysis
The final tab provides detailed sentiment analysis and visualizations:


###Installation###
Prerequisites
To run this app, you'll need:

Python 3.8+

Streamlit and other required Python packages (listed below).

Setup Instructions

Clone the repository:
git clone https://github.com/Uplyuz/PolarWeb.git
Install the dependencies:
streamlit
requests
pandas
matplotlib
wordcloud
deep_translator
transformers
(see the requierements. txt, requierements might change or be replaced) 

Obtain Twitter API access:
Create a RapidAPI account and subscribe to the Twitter API.
Obtain your API keys and set them up in the script (or as environment variables).

Run the app: Launch the app using the following command:
streamlit run app.py
This will start a local server.

Additional Scripts and Functionality
translate_API_output.py
This script includes functions (traducir, traducir_tweets) that handle the translation of tweet texts into a specified language using the Deep Translator library.

custom_class_final_model.py
This script contains the custom fine-tuned RoBERTa model for performing sentiment analysis. The model is loaded and applied to the tweets to classify them as Positive or Negative.

model_load_apply.py
This script includes functions to load the custom model and apply it to the tweet data. The key functions include:

load_custom_sentiment_model(): Loads the fine-tuned RoBERTa model and tokenizer.
predict_sentiment(): Predicts the sentiment of individual tweets.
analyze_sentiments(): Applies the model to a dataset of tweets and returns sentiment labels (0 for Negative, 1 for Positive).
dashboard_charts.py
This script is used to generate the word cloud for visualizing the most common words in positive and negative tweets.

##Please find in the folder named "Model tests and train" all informations abuot the model training and metrics achieved. 

###Example Usage###
Set Up Your Search: Enter a keyword and choose between "Top" or "Latest" tweets.
Get Data: After performing the search, view the retrieved tweets with sentiment analysis in tabular form and download the data if needed.
Get Analysis: View the sentiment analysis results through interactive bar charts and word clouds, along with additional insights like total likes on the tweets.

Acknowledgements
Streamlit for providing the framework for building this interactive web application.
Hugging Face for dataset used to fine tune the RoBERTa model. 
RapidAPI for the Twitter API used to retrieve tweets.
Deep Translator for translating tweet content.
4 Geeks Accademy for the teachings during this course. 



