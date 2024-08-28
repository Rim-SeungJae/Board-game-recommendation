import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from tqdm import tqdm
import dask.dataframe as dd
import dask.array as da
import sys
from collections import OrderedDict
import ast


def parse_suggested_num_players(suggested_data_str):
    try:
        # 문자열 형태의 데이터를 안전하게 파싱
        suggested_data = eval(suggested_data_str)

        parsed_data = []
        for entry in suggested_data:
            best_votes = 0
            recommended_votes = 0
            if isinstance(entry, dict):
                for result in entry.get('result', []):
                    if result['@value'] == 'Best':
                        best_votes = int(result['@numvotes'])
                    elif result['@value'] == 'Recommended':
                        recommended_votes = int(result['@numvotes'])
            total_votes = best_votes + recommended_votes
            parsed_data.append(total_votes)
        return sum(parsed_data)  # 모든 플레이 인원수에 대해 합산된 투표 수 반환
    except (SyntaxError, ValueError) as e:
        # 오류가 발생할 경우 0을 반환
        return 0

def save_content_based_similarity():
    # CSV 파일 로드
    bg_info = pd.read_csv('./archive/games_detailed_info.csv')

    # 필요한 컬럼들 선택
    bg_info = bg_info[['primary', 'minplayers', 'maxplayers', 'minplaytime', 'maxplaytime', 'boardgamecategory', 'boardgamemechanic', 'suggested_num_players']]

    # 결측치 제거
    bg_info = bg_info.dropna()

    # 문자열을 리스트로 변환
    bg_info['boardgamecategory'] = bg_info['boardgamecategory'].apply(lambda x: " ".join(eval(x)))
    bg_info['boardgamemechanic'] = bg_info['boardgamemechanic'].apply(lambda x: " ".join(eval(x)))

    # suggested_num_players 필드 파싱
    bg_info['suggested_num_players'] = bg_info['suggested_num_players'].apply(parse_suggested_num_players)

    # 카테고리 및 메커니즘에 대한 벡터화
    vectorizer = CountVectorizer()
    category_matrix = vectorizer.fit_transform(bg_info['boardgamecategory'].to_numpy())
    mechanic_matrix = vectorizer.fit_transform(bg_info['boardgamemechanic'].to_numpy())

    # 플레이 인원수 및 플레이 시간 필드
    minplayers_matrix = bg_info['minplayers'].to_numpy().reshape(-1, 1)
    maxplayers_matrix = bg_info['maxplayers'].to_numpy().reshape(-1, 1)
    minplaytime_matrix = bg_info['minplaytime'].to_numpy().reshape(-1, 1)
    maxplaytime_matrix = bg_info['maxplaytime'].to_numpy().reshape(-1, 1)

    # 가중치 적용
    suggested_players_matrix = bg_info['suggested_num_players'].to_numpy().reshape(-1, 1)

    # 모든 매트릭스 병합
    combined_matrix = sparse.hstack((
        category_matrix,
        mechanic_matrix,
        minplayers_matrix,
        maxplayers_matrix,
        minplaytime_matrix,
        maxplaytime_matrix,
        suggested_players_matrix
    ))

    # 코사인 유사도 계산
    similarity = cosine_similarity(combined_matrix, combined_matrix)

    # 결과 저장
    np.save('similarity.npy', similarity)



def save_corr_matrix():
    ################################################
    # input:
    # output: saves correlation matrix to corr_matrix.npy
    ################################################
    bg_rating = pd.read_csv('./archive/bgg-15m-reviews.csv')

    # drop rows(due to memory lacking issue)
    bg_rating = bg_rating[:-int(bg_rating.shape[0]/4)]

    bg_rating.drop('comment', axis=1, inplace=True)

    chunk_size = 500000
    chunks = [x for x in range(0, bg_rating.shape[0], chunk_size)]
    chunks.append(bg_rating.shape[0])
    bg_rating_pivot = pd.DataFrame(dtype=np.float16)
    for i in tqdm(range(0, len(chunks)-1)):
        bg_rating_chunk = bg_rating.iloc[chunks[i]:chunks[i+1] - 1]
        pivot_chunk = (bg_rating_chunk.groupby(['user', 'name'])['rating']
                       .sum()
                       .unstack()
                       .reset_index()
                       .set_index('user')
                       )
        pivot_chunk = pivot_chunk.astype(np.float16)
        bg_rating_pivot = bg_rating_pivot.append(pivot_chunk, sort=False)
        # if bg_rating_pivot.empty:
        #     bg_rating_pivot = bg_rating_pivot.append(pivot_chunk, sort=False)
        # else:
        #     bg_rating_pivot = pd.merge(bg_rating_pivot, pivot_chunk, on='user', how='outer')
        bg_rating_pivot = bg_rating_pivot.groupby('user').sum().reset_index().set_index('user')
        # print(np.nonzero(bg_rating_pivot.columns.duplicated()))

    # bg_rating_sparse = sparse.csr_matrix(bg_rating_pivot.fillna(0).to_numpy())
    bg_user_matrix = bg_rating_pivot.T
    SVD = TruncatedSVD(n_components=20)
    corrcoef_matrix = SVD.fit_transform(bg_user_matrix)
    corr_matrix = np.corrcoef(corrcoef_matrix)

    title = bg_rating_pivot.columns
    title_array = np.array(list(title))

    np.save('corr_matrix.npy', corr_matrix)
    np.save('bg_titles.npy', title_array)



if __name__ == '__main__':
    save_content_based_similarity()
    #save_corr_matrix()
