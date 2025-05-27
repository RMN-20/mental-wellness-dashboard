import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    vader_scores = vader_analyzer.polarity_scores(text)
    compound = vader_scores['compound']

    return {
        'TextBlob Polarity': polarity,
        'VADER Compound': compound,
        'VADER Positive': vader_scores['pos'],
        'VADER Negative': vader_scores['neg'],
        'VADER Neutral': vader_scores['neu']
    }

def analyze_journal_csv(csv_path):
    df = pd.read_csv(csv_path)
    sentiments = df['entry'].apply(analyze_sentiment)
    sentiment_df = pd.DataFrame(sentiments.tolist())
    result = pd.concat([df, sentiment_df], axis=1)
    return result
