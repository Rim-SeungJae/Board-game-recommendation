import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

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

bg_rating = pd.read_csv('C:/Users/dipreez/Desktop/졸작/Board-game-recommendation/project/archive/bgg-15m-reviews.csv', on_bad_lines='skip',low_memory=False)
bg_rating.drop('comment', axis=1, inplace=True)
bg_rating = bg_rating[['user', 'name', 'rating','ID']].dropna()

num_users = len(bg_rating['user'].unique())
num_items = len(bg_rating['name'].unique())
embedding_dim = 20
hidden_layers = [64, 32]

# 모델 로드
model = NCF(num_users, num_items, embedding_dim, hidden_layers)
model.load_state_dict(torch.load('ncf_model.pth'))
model.eval()

def recommend_similar_games(game_name, model, bg_rating, top_n=10):
    # 보드게임의 아이디 가져오기
    item_id = bg_rating['name'].astype('category').cat.codes[bg_rating['name'] == game_name].values[0]
    print()

    # 모든 보드게임에 대한 예측 평점 계산
    user_id = torch.tensor([0])  # dummy user
    item_ids = torch.tensor(list(range(num_items)))

    inputs = torch.stack([user_id.repeat(num_items), item_ids], dim=1)
    predictions = model(inputs).detach().numpy()

    # 예측 평점을 기준으로 유사한 보드게임 추천
    top_indices = predictions.argsort()[-top_n:][::-1]

    # 추천된 보드게임 목록 반환
    recommended_games = bg_rating['name'].astype('category').cat.categories[top_indices]
    return recommended_games

while True:
    print("input: ")
    name = input()
    # 예시로 'Catan'과 비슷한 보드게임 추천
    recommended_games = recommend_similar_games(name, model, bg_rating)
    print(recommended_games)
