import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from tqdm import tqdm
import sys

def save_similarity():
    ################################################
    # input:
    # output: saves similarity matrix calculated by content-based filtering to similarity.npy
    ################################################
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

def save_corrcoef():
    ################################################
    # input:
    # output: saves correlation coefficient to corrcoef.npy
    ################################################
    bg_rating = pd.read_csv('./archive/bgg-15m-reviews.csv')

    # randomly drop rows(due to memory lacking issue)
    drop_indices = np.random.choice(bg_rating.index, int(bg_rating.shape[0]/2), replace=False)

    bg_rating.drop(drop_indices, inplace=True)
    bg_rating.drop('comment', axis=1, inplace=True)

    chunk_size = 500000
    chunks = [x for x in range(0, bg_rating.shape[0], chunk_size)]
    chunks.append(bg_rating.shape[0])
    print(chunks)
    bg_rating_pivot = pd.DataFrame(dtype=np.float16)
    for i in tqdm(range(0, len(chunks))):
        bg_rating_chunk = bg_rating.iloc[chunks[i]:chunks[i+1] - 1]
        pivot_chunk = (bg_rating_chunk.groupby(['user', 'name'])['rating']
                       .sum()
                       .unstack()
                       .reset_index()
                       .set_index('user')
                       )
        pivot_chunk = pivot_chunk.astype(np.float16)
        if bg_rating_pivot.empty:
            bg_rating_pivot = bg_rating_pivot.append(pivot_chunk, sort=False)
            print('hi')
        else:
            bg_rating_pivot = pd.merge(bg_rating_pivot, pivot_chunk, on='user', how='outer')
        # bg_rating_pivot = bg_rating_pivot.groupby('user').sum().reset_index().set_index('user')
        # print(np.nonzero(bg_rating_pivot.columns.duplicated()))

    bg_rating_sparse = sparse.csr_matrix(bg_rating_pivot.fillna(0).to_numpy())
    print(bg_rating_sparse)
    np.save('bg_rating_sparse.npy', bg_rating_sparse)

if __name__ == '__main__':
    # save_similarity()
    save_corrcoef()