import pandas as pd
from textblob import TextBlob
import os

# Check if the dataset file exists
file_path = 'D://AIDI//AIDI 1006//Assignment 4//data//cleaned_video_game_data.csv'
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' does not exist in the current directory.")
else:
    # Load your dataset
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check if the dataset is loaded correctly
    print(f"Loaded dataset with {len(df)} rows.")
    
    # Check the first few rows to ensure the 'User Review Text' column is present
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    # Function to perform sentiment analysis using TextBlob
    def get_sentiment(text):
        if pd.isna(text):  # Check for NaN values
            return 0
        blob = TextBlob(text)
        return blob.sentiment.polarity  # Return polarity (ranges from -1 to 1)

    # Apply the sentiment function to the 'User Review Text' column
    print("\nPerforming sentiment analysis...")
    df['Sentiment'] = df['User Review Text'].apply(get_sentiment)
    
    # Classify sentiment as positive, neutral, or negative based on polarity score
    def classify_sentiment(polarity):
        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    # Apply sentiment classification
    df['Sentiment Label'] = df['Sentiment'].apply(classify_sentiment)

    # Check the results after sentiment analysis
    print("\nSentiment analysis complete. First few rows of the results:")
    print(df[['Game Title', 'User Review Text', 'Sentiment', 'Sentiment Label']].head())

    # Save the results with sentiment labels into a new CSV file
    output_file = 'D://AIDI//AIDI 1006//Assignment 4//data//video_game_reviews_with_sentiment.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nSentiment analysis complete. Results saved to '{output_file}'.")
