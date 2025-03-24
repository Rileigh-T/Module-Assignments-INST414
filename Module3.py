import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('all_video_games(cleaned).csv')
df['User Score'] = df['User Score'].fillna(0)  
df['User Ratings Count'] = df['User Ratings Count'].fillna(0) 
df['Genres'] = df['Genres'].fillna('Unknown')

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(df['Genres'])

numerical_features = df[['User Score', 'User Ratings Count']]
numerical_features = (numerical_features - numerical_features.mean()) / numerical_features.std()


final = np.hstack((genre_matrix.toarray(), numerical_features))
cosine_sim = cosine_similarity(final)

def get_most_similar_games(game_index, top_n=10):
    similarity_scores = list(enumerate(cosine_sim[game_index]))
    sorted_similar_games = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_games = sorted_similar_games[1:top_n+1]
    similar_game_indices = [game[0] for game in top_similar_games]
    return df.iloc[similar_game_indices][['Title', 'Genres', 'User Score', 'User Ratings Count']]

def get_game_index_by_title(game_title):
    return df[df['Title'] == game_title].index[0]

sims_index = get_game_index_by_title("The Sims 2")
similar_sims_games = get_most_similar_games(sims_index, top_n=10)

rainbow_six_index = get_game_index_by_title("Tom Clancy's Rainbow Six Siege")
similar_rainbow_six_games = get_most_similar_games(rainbow_six_index, top_n=10)

horizon_zero_dawn_index = get_game_index_by_title("Horizon Zero Dawn")
similar_horizon_zero_dawn_games = get_most_similar_games(horizon_zero_dawn_index, top_n=10)

def plot_table(data, title):
    fig, ax = plt.subplots(figsize=(8, 4)) 
    table = ax.table(cellText=data.values, colLabels=data.columns, loc='center', cellLoc='center')
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.show()

plot_table(similar_sims_games, "Similar Games to 'The Sims 2'")
plot_table(similar_rainbow_six_games, "Similar Games to 'Rainbow Six Siege'")
plot_table(similar_horizon_zero_dawn_games, "Similar Games to 'Horizon Zero Dawn'")
