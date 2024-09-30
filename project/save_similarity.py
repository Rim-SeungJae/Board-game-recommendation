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

import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_layers):
        super(NCF, self).__init__()
        # GMF part
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

        # MLP part
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        self.mlp_layers = nn.Sequential()
        input_dim = 2 * embedding_dim

        self.mlp_layers.add_module("linear_0", nn.Linear(input_dim, hidden_layers[0]))
        self.mlp_layers.add_module("relu_0", nn.ReLU())

        for i, hidden_dim in enumerate(hidden_layers[1:], start=1):
            self.mlp_layers.add_module(f"linear_{i}", nn.Linear(hidden_layers[i-1], hidden_dim))
            self.mlp_layers.add_module(f"relu_{i}", nn.ReLU())

        # Combine GMF and MLP
        self.final_linear = nn.Linear(hidden_layers[-1] + embedding_dim, 1)

    def forward(self, x):
        user_id = x[:, 0]
        item_id = x[:, 1]

        # GMF part
        gmf_user_embedding = self.user_embedding_gmf(user_id)
        gmf_item_embedding = self.item_embedding_gmf(item_id)
        gmf_output = gmf_user_embedding * gmf_item_embedding

        # MLP part
        mlp_user_embedding = self.user_embedding_mlp(user_id)
        mlp_item_embedding = self.item_embedding_mlp(item_id)
        mlp_input = torch.cat([mlp_user_embedding, mlp_item_embedding], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        # Combine GMF and MLP outputs
        concat_output = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = torch.sigmoid(self.final_linear(concat_output))

        return prediction.squeeze()


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

def save_top_k_from_existing_corr_matrix(npy_file_path, k=10, output_file_path='top_k_corr_matrix.npy'):
    ################################################
    # input:
    # npy_file_path: path to existing correlation matrix npy file
    # k: number of top similar items to retain for each item
    # output: saves top-k correlation matrix to a new npy file
    ################################################

    # Load existing correlation matrix
    corr_matrix = np.load(npy_file_path)

    # Initialize matrix to store top-k similarities
    top_k_corr_matrix = np.zeros_like(corr_matrix)

    # Retain only top-k similarities for each item
    for i in range(corr_matrix.shape[0]):
        top_k_indices = np.argsort(-corr_matrix[i])[1:k+1]  # Retain top k+1 (including self)
        top_k_corr_matrix[i, top_k_indices] = corr_matrix[i, top_k_indices]

    # Save the modified matrix
    np.save(output_file_path, top_k_corr_matrix)

# NCF 모델의 GMF와 MLP 아이템 임베딩을 결합하여 유사도 행렬 계산
def save_ncf_corr_matrix():

    model = NCF(num_users=351048, num_items=18984, embedding_dim=20, hidden_layers=[64, 32])
    gmf_item_embedding = model.item_embedding_gmf.weight.detach().cpu().numpy()
    mlp_item_embedding = model.item_embedding_mlp.weight.detach().cpu().numpy()

    combined_item_embedding = np.concatenate((gmf_item_embedding, mlp_item_embedding), axis=1)
    corr_matrix = np.corrcoef(combined_item_embedding)

    model.load_state_dict(torch.load('C:/Users/dipreez/Desktop/졸작/Board-game-recommendation/project/ncf_model.pth'))
    model.eval()

    print(np.shape(corr_matrix))

    np.save('ncf_corr_matrix.npy', corr_matrix)


def save_ncf_top_k_similarity(k=10, save_path="ncf_top_k_similarity.npy"):
    model = NCF(num_users=351048, num_items=18984, embedding_dim=20, hidden_layers=[64, 32])
    model.load_state_dict(torch.load('C:/Users/dipreez/Desktop/졸작/Board-game-recommendation/project/ncf_model.pth'))
    model.eval()

    gmf_item_embedding = model.item_embedding_gmf.weight.detach().cpu().numpy()
    mlp_item_embedding = model.item_embedding_mlp.weight.detach().cpu().numpy()

    combined_item_embedding = np.concatenate((gmf_item_embedding, mlp_item_embedding), axis=1)

    num_items = combined_item_embedding.shape[0]
    top_k_similarities = {}

    for i in range(num_items):
        # i번째 아이템의 임베딩 벡터
        target_vector = combined_item_embedding[i]

        # 모든 아이템과의 코사인 유사도 계산
        similarity_scores = np.dot(combined_item_embedding, target_vector) / (
                np.linalg.norm(combined_item_embedding, axis=1) * np.linalg.norm(target_vector) + 1e-9
        )

        # 자신을 제외한 상위 k개의 유사 아이템 선택
        top_k_indices = np.argsort(-similarity_scores)[1:k+1]
        top_k_values = similarity_scores[top_k_indices]

        # 딕셔너리에 인덱스와 해당 유사도를 저장
        top_k_similarities[i] = list(zip(top_k_indices, top_k_values))

    np.save(save_path, top_k_similarities)
    print(f"Top-{k} NCF similarity matrix saved to {save_path}")



if __name__ == '__main__':
    #save_content_based_similarity()
    #save_corr_matrix()
    save_top_k_from_existing_corr_matrix('C:/Users/dipreez/Desktop/졸작/Board-game-recommendation/project/corr_matrix.npy', k=10)
