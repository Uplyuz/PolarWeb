# Tab 4: Heads Up - Keyword Comparison
with tab4:
    st.markdown("<h3 style='text-align: center;'>Compare Sentiment for Two Keywords</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        keyword1 = st.text_input("Enter first keyword to search tweets:", "Keyword 1")
    with col2:
        keyword2 = st.text_input("Enter second keyword to search tweets:", "Keyword 2")
    
    tweet_limit = 20
    
    if st.button("Compare"):
        # Check if the keywords are valid
        if not keyword1.strip() or not keyword2.strip():
            st.warning("Please enter both keywords before comparing.")
        else:
            # Prepare query strings for each keyword
            querystring1 = {"query": keyword1, "section": "latest", "limit": (tweet_limit)}
            querystring2 = {"query": keyword2, "section": "latest", "limit": (tweet_limit)}
            
            try:
                # Fetch tweets for each keyword using pagination function
                tweets1 = fetch_tweets_with_pagination(url_tweets_search_api_01, querystring1, headers)
                tweets2 = fetch_tweets_with_pagination(url_tweets_search_api_01, querystring2, headers)
                
                # Convert fetched tweets to DataFrames
                search1 = pd.DataFrame(tweets1, columns=['Date', 'Tweet', 'Tweet_Likes'])
                search2 = pd.DataFrame(tweets2, columns=['Date', 'Tweet', 'Tweet_Likes'])
                
                # Preprocess and translate tweets for each search
                search1 = preprocess_tweet(search1)
                search1['Tweet'] = search1['Tweet'].apply(traducir)
                search1 = search1.dropna(subset=['Tweet'])
                search1 = search1[search1['Tweet'].str.strip().astype(bool)]
                
                search2 = preprocess_tweet(search2)
                search2['Tweet'] = search2['Tweet'].apply(traducir)
                search2 = search2.dropna(subset=['Tweet'])
                search2 = search2[search2['Tweet'].str.strip().astype(bool)]
                
                # Apply sentiment analysis using the model on both search results
                search1 = analyze_sentiments(model_custom, tokenizer_custom, search1)
                search2 = analyze_sentiments(model_custom, tokenizer_custom, search2)
                
                st.success("Sentiment analysis completed for both keywords.")
                
                # Display sentiment distribution for the first keyword
                st.markdown(f"### Sentiment Distribution for **{keyword1}**")
                sentiment_dist_plotly(search1)  # Plot distribution for the first keyword
                
                st.markdown(f"### Sentiment Distribution for **{keyword2}**")
                sentiment_dist_plotly(search2)  # Plot distribution for the second keyword
                
            except Exception as e:
                st.error(f"An error occurred during the comparison: {e}")


#graph to add: bar charts with comparison bars