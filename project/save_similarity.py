import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def save_similarity():
    bg_info = pd.read_csv('./archive/games_detailed_info.csv')
    bg_info = bg_info[['primary', 'boardgamecategory']]
    bg_info = bg_info.dropna()

    bg_info['boardgamecategory'] = bg_info['boardgamecategory'].apply(literal_eval)
    bg_info['boardgamecategory'] = bg_info['boardgamecategory'].apply(lambda x : " ".join(x))

    vectorizer = CountVectorizer()
    category_vector = vectorizer.fit_transform(bg_info['boardgamecategory'].to_numpy())

    similarity = cosine_similarity(category_vector, category_vector).argsort()[:, ::-1]

    np.save('similarity.npy', similarity)

if __name__ == '__main__':
    save_similarity()