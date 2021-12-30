import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_k_recommendation(board_game_title, k=10):
    bg_info = pd.read_csv('./archive/games_detailed_info.csv')
    bg_info = bg_info[['primary', 'minplayers', 'maxplayers', 'boardgamecategory', 'boardgamemechanic']]
    bg_info = bg_info.dropna()
    similarity = np.load('./similarity.npy')

    target_idx = bg_info[bg_info['primary'] == board_game_title].index.values

    similar_idx = similarity[target_idx, :k].reshape(-1)

    similar_idx = similar_idx[similar_idx != target_idx]

    recommendation = bg_info.iloc[similar_idx]

    print(recommendation)

    return recommendation


if __name__ == '__main__':
    get_k_recommendation("Pandemic")