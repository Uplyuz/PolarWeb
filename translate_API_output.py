from deep_translator import GoogleTranslator
import re
#df_prueba_roberta.rename(columns={'sentiment': 'tweets'}, inplace=True)

def traducir(text):
    # Detectar el idioma automáticamente y traducirlo al inglés
    try:
        translator = GoogleTranslator(source='auto', target='en')
        translated_text=translator.translate(text)
        return translated_text
    
    except Exception as e:

        print(f"Error al traducir el texto {e}")
        return text
    
def traducir_tweets(df):

    # Asegúrate de que la columna 'tweets' existe
    if 'Tweet' in df.columns:
        df['Translated_tweet'] = df['Tweet'].apply(traducir)
    else:
        print("La columna 'Tweet' no se encontró en el DataFrame.")
    return df

# df_prueba_roberta['tweets_traducidos'] = df_prueba_roberta['tweets'].apply(traducir)


def preprocess_tweet(df):
    df['Tweet'] = df['Tweet'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x))  # delete URLs
    df['Tweet'] = df['Tweet'].apply(lambda x: re.sub(r'@\w+', '', x))  # delete mentions
    #df['Tweet'] = df['Tweet'].apply(lambda x: re.sub(r'#\w+', '', x))  # delete hashtags
    df_clean_data = df.copy()
    return df_clean_data