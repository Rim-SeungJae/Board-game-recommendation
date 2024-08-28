import pandas as pd
import numpy as np

def get_content_based_reccomendation(board_game_title, k=10):
    ################################################
    # input: gets title of a board game, k value
    # output: returns list of top k board games that are similar to the input board game
    ################################################
    bg_info = pd.read_csv('./archive/games_detailed_info.csv')
    bg_info = bg_info[['primary', 'minplayers', 'maxplayers', 'boardgamecategory', 'boardgamemechanic']]
    bg_info = bg_info.dropna()
    similarity = np.load('./similarity.npy')

    target_idx = bg_info[bg_info['primary'] == board_game_title].index.values

    similar_idx = similarity[target_idx, :k+1].reshape(-1)

    similar_idx = similar_idx[similar_idx != target_idx]

    recommendation = bg_info.iloc[similar_idx]
    recommendation = recommendation['primary'].tolist()

    print(recommendation)

    return recommendation

def get_colaborative_filtering_recommendation(board_game_title, k=10):
    ################################################
    # input: gets title of a board game, k value
    # output: returns list of top k board games that are similar to the input board game
    ################################################
    bg_titles = np.load('./bg_titles.npy')
    corr_matrix = np.load('./corr_matrix.npy')
    corr_matrix = corr_matrix.argsort()[:, ::-1]
    bg_titles_list = list(bg_titles)

    target_idx = bg_titles_list.index(board_game_title)

    similar_idx = corr_matrix[target_idx, :k+1].reshape(-1)

    similar_idx = similar_idx[similar_idx != target_idx]

    recommendation = list(bg_titles[similar_idx])

    print(recommendation)

    return recommendation

if __name__ == '__main__':
    get_content_based_reccomendation("Pandemic")
    get_colaborative_filtering_recommendation("Pandemic")
