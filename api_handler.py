import requests
import time

# Pagination function with API URL passed as a parameter
def fetch_tweets_with_pagination(url, querystring, headers, max_tweets=50):
    all_entries = []
    page_number = 1
    last_timestamp = None
    retries = 3  # Number of retries if a request fails

    while len(all_entries) < max_tweets:
        print(f"Fetching Page {page_number}...")

        # Timestamp-based pagination: Fetch older tweets before the last timestamp
        if last_timestamp:
            querystring['until'] = last_timestamp

        # Make the API request with retry logic
        for attempt in range(retries):
            response = requests.get(url, headers=headers, params=querystring)  # URL now passed in here
            if response.status_code == 200:
                break
            else:
                print(f"Error: {response.status_code}, Retrying {attempt + 1}/{retries}...")
                time.sleep(1)  # Wait before retrying

        if response.status_code != 200:
            print(f"Failed to fetch page {page_number}: {response.status_code}")
            break

        # Parse the response
        data = response.json()

        # Check if 'data' exists in the response, else log the issue
        if 'data' not in data:
            print(f"Unexpected response structure on Page {page_number}: {data}")
            break

        # Extract entries and process
        try:
            timeline = data['data']['search_by_raw_query']['search_timeline']['timeline']
            entries = timeline['instructions'][0]['entries']
        except (KeyError, IndexError) as e:
            print(f"Error parsing entries on Page {page_number}: {e}")
            break

        # Call the cleaning function (assumed to be part of this module or imported)
        cleaned_entries = clean_entries_with_dates(entries)
        all_entries.extend(cleaned_entries)

        # Update last timestamp for timestamp-based pagination
        if cleaned_entries:
            last_timestamp = cleaned_entries[-1][1]  # Update to the last tweet's timestamp

        # Stop if no more entries are found
        if not entries or len(entries) == 0:
            print("No more tweets to fetch.")
            break

        # Increment page number for pagination tracking
        page_number += 1

    return all_entries[:max_tweets]  # Return the collected tweets

# Cleaning function (already in your api_handler.py)
def clean_entries_with_dates(list_of_elem):
    clean_data = []
    for element in list_of_elem:
        content = element.get('content', [])  # Extract 'content'
        item_content = content.get('itemContent', {})  # Extract 'itemContent'
        
        if 'tweet_results' not in item_content or 'result' not in item_content['tweet_results']:
            continue
        
        result = item_content['tweet_results']['result']
        if 'legacy' not in result or 'full_text' not in result['legacy']:
            continue
        
        # Extract full_text, post_date, and favorite_count (likes)
        full_text = result['legacy']['full_text']
        post_date = result['legacy']['created_at']
        favorite_count = result['legacy'].get('favorite_count', 0)  # Get 'favorite_count', defaulting to 0 if missing
        
        clean_data.append((post_date, full_text, favorite_count))  # Add likes to the tuple
    
    return clean_data
