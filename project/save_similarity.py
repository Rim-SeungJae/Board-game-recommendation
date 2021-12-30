import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

def save_similarity():
    bg_info = pd.read_csv('./archive/games_detailed_info.csv')
    bg_info = bg_info[['primary', 'minplayers', 'maxplayers', 'boardgamecategory', 'boardgamemechanic']]
    bg_info = bg_info.dropna()

    bg_info['boardgamecategory'] = bg_info['boardgamecategory'].apply(literal_eval)
    bg_info['boardgamecategory'] = bg_info['boardgamecategory'].apply(lambda x : " ".join(x))
    bg_info['boardgamemechanic'] = bg_info['boardgamemechanic'].apply(literal_eval)
    bg_info['boardgamemechanic'] = bg_info['boardgamemechanic'].apply(lambda x : " ".join(x))

    vectorizer = CountVectorizer()
    category_matrix = vectorizer.fit_transform(bg_info['boardgamecategory'].to_numpy())
    mechanic_matrix = vectorizer.fit_transform(bg_info['boardgamemechanic'].to_numpy())

    combined_matrix = sparse.hstack((category_matrix, mechanic_matrix, bg_info['minplayers'].to_numpy().reshape(-1,1), bg_info['maxplayers'].to_numpy().reshape(-1,1)))

    similarity = cosine_similarity(combined_matrix, combined_matrix).argsort()[:, ::-1]

    np.save('similarity.npy', similarity)

if __name__ == '__main__':
    save_similarity()