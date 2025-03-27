# Video Game Sentiment Analysis and Recommendation System

This project involves analyzing sentiment in video game reviews and recommending similar games based on user input. The system processes user reviews and assigns sentiment labels (positive, negative, or neutral) to the games. It then utilizes a recommendation system to suggest similar games based on the text content of the reviews.

## Project Structure

The project consists of the following steps:

1. **Sentiment Analysis:** 
   - The sentiment of user reviews is analyzed using the `TextBlob` library to classify them as positive, neutral, or negative.
   - The sentiment polarity score ranges from -1 (negative) to 1 (positive).

2. **Recommendation System:**
   - A content-based recommendation system is implemented using the `TF-IDF` vectorizer to convert reviews into numerical features.
   - Dimensionality reduction is performed using `TruncatedSVD` to improve the performance of the recommendation system.
   - Cosine similarity is computed between the games based on their review content to recommend similar games.

3. **Data Visualization:**
   - The distribution of sentiment labels is visualized using a count plot.
   - The relationship between user ratings and sentiment labels is visualized using a boxplot.
   - A word cloud is generated from all the user reviews to highlight the most frequent words.

## Dataset

The dataset used for this project includes information about video games, their user ratings, reviews, and sentiment labels. It contains the following columns:

- `Game Title`: The title of the game.
- `User Rating`: The rating given by users.
- `Age Group`: The targeted age group for the game.
- `Price`: The price of the game.
- `Platform`: The platform on which the game is available.
- `Developer`: The developer of the game.
- `Publisher`: The publisher of the game.
- `Release Year`: The year the game was released.
- `Genre`: The genre of the game.
- `User Review Text`: The text of the user review.
- `Sentiment Label`: The sentiment label assigned to the review (Positive, Neutral, Negative).
- `Polarity`: The numerical polarity value of the sentiment.

## Dependencies

To run this project, you need to install the following Python libraries:

- `pandas`
- `matplotlib`
- `seaborn`
- `textblob`
- `sklearn`
- `wordcloud`
- `scipy`

You can install these dependencies using `pip`:

```bash
pip install pandas matplotlib seaborn textblob scikit-learn wordcloud scipy
