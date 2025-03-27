from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('D://AIDI//AIDI 1006//Assignment 4//data//video_game_reviews_with_sentiment_and_polarity.csv')

# Clean the dataset by dropping rows with missing content
df = df.dropna(subset=['User Review Text'])

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the content column to get the TF-IDF features
tfidf_matrix = vectorizer.fit_transform(df['User Review Text'])

# Print TF-IDF matrix shape
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# Apply Truncated SVD to reduce dimensionality (set n_components to 5, which is <= 9 features)
svd = TruncatedSVD(n_components=5, random_state=42)  # Adjusted to a number <= 9
reduced_matrix = svd.fit_transform(tfidf_matrix)

# Compute cosine similarity on the reduced matrix
cosine_sim = cosine_similarity(reduced_matrix, reduced_matrix)

# Function to recommend games based on similarity
def recommend_game(game_title, cosine_sim=cosine_sim):
    # Find the index of the game that matches the title
    idx = df[df['Game Title'].str.contains(game_title, case=False)].index[0]
    
    # Get the pairwise similarity scores for all games with the input game
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the games based on the similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top 5 most similar games (excluding the input game itself)
    sim_scores = sim_scores[1:6]
    
    # Get the game indices
    game_indices = [i[0] for i in sim_scores]
    
    # Return the top 5 most similar games
    recommended_games = df.iloc[game_indices][['Game Title', 'Sentiment Label', 'Polarity']]
    
    return recommended_games

# Example: Get recommendations for a game
user_game = input("Enter the name of the game you've liked: ")
recommended_games = recommend_game(user_game)

print("\nRecommended games based on your preference:\n")
print(recommended_games)
