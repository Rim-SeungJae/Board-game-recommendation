import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    bg_info = pd.read_csv('./archive/games_detailed_info.csv')
    bg_info = bg_info[['description', 'minplayers', 'maxplayers', 'playingtime', 'boardgamecategory', 'boardgamemechanic', 'boardgamedesigner', 'boardgameartist', 'boardgamepublisher', 'bayesaverage']]
    bg_info = bg_info.dropna()

    bg_info['boardgamecategory'] = bg_info['boardgamecategory'].apply(literal_eval)
    bg_info['boardgamemechanic'] = bg_info['boardgamemechanic'].apply(literal_eval)
    bg_info['boardgamedesigner'] = bg_info['boardgamedesigner'].apply(literal_eval)
    bg_info['boardgameartist'] = bg_info['boardgameartist'].apply(literal_eval)
    bg_info['boardgamepublisher'] = bg_info['boardgamepublisher'].apply(literal_eval)

    bg_info['boardgamecategory'] = bg_info['boardgamecategory'].apply(lambda x : " ".join(x))
    vectorizer = TfidfVectorizer()
    category_vector = vectorizer.fit_transform(bg_info['boardgamecategory'])

    similarity = cosine_similarity(category_vector, category_vector).argsort()[:, ::-1]