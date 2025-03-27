from django.views.generic import ListView, DetailView
from django.views.generic.dates import ArchiveIndexView, YearArchiveView,MonthArchiveView
from django.views.generic.dates import DayArchiveView, TodayArchiveView

from list.models import Boardgame, Boardgame_detail, Rating
from django.conf import settings
from django.views.generic import FormView
from django.db.models import Q
from django.shortcuts import render

from django.views.generic import CreateView,UpdateView,DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from mysite.views import OwnerOnlyMixin

import pandas as pd
import numpy as np
import os
from django.core.cache import cache
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from .forms import BoardgameSearchForm

import torch
import torch.nn as nn
import torch.nn.functional as F

import random


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

NCF_model = NCF(351048,18984,20,[64,32])
model_load_path = os.path.join(settings.MEDIA_ROOT, 'models', 'ncf_model.pth')
NCF_model.load_state_dict(torch.load(model_load_path,map_location=torch.device('cpu')))

class BoardgameLV(ListView):
    model=Boardgame
    template_name='list/post_all.html'
    context_object_name='boardgames'
    paginate_by=18

# Load the npy files
similarity_path = os.path.join(settings.MEDIA_ROOT, 'npys', 'similarity.npy')
corr_matrix_path = os.path.join(settings.MEDIA_ROOT, 'npys', 'top_k_corr_matrix.npy')
ncf_corr_matrix_path = os.path.join(settings.MEDIA_ROOT, 'npys', 'ncf_top_k_similarity.npy');
bg_titles_path = os.path.join(settings.MEDIA_ROOT, 'npys', 'bg_titles.npy')

bg_titles = np.load(bg_titles_path)
similarity = np.load(similarity_path)
top_k_corr_matrix = np.load(corr_matrix_path)
# corr_matrix = corr_matrix.argsort()[:, ::-1]
ncf_corr_matrix = np.load(ncf_corr_matrix_path, allow_pickle=True).item()

mapping_table_path = os.path.join(settings.MEDIA_ROOT, 'npys', 'map.npy')
mapping_table = np.load(mapping_table_path, allow_pickle=True)
mapping_table = pd.DataFrame(mapping_table)

def recommend_for_new_user(user_ratings, top_k=10):

    gmf_item_embedding = NCF_model.item_embedding_gmf.weight.detach().cpu().numpy()
    mlp_item_embedding = NCF_model.item_embedding_mlp.weight.detach().cpu().numpy()
    item_embeddings = np.concatenate((gmf_item_embedding, mlp_item_embedding), axis=1)

    profile_vectors = []
    scores = []


    for db_index, rating in user_ratings:
        model_index = mapping_table[mapping_table.iloc[:,0] == db_index][1].iloc[0]
        print(model_index)
        profile_vectors.append(item_embeddings[model_index])
        scores.append(rating)

    if not profile_vectors:
        return []

    profile_vector = np.average(profile_vectors, axis=0, weights=scores)

    profile_norm = np.linalg.norm(profile_vector)
    item_norms = np.linalg.norm(item_embeddings, axis=1)
    dot_products = np.dot(item_embeddings, profile_vector)
    similarities = dot_products / (item_norms * profile_norm + 1e-8)

    rated_model_indices = [mapping_table[mapping_table.iloc[:,0] == db][1].iloc[0]
                           for db, _ in user_ratings]

    top_indices = np.argsort(-similarities)[:top_k]
    print(top_indices)
    recommendation = [mapping_table[mapping_table.iloc[:,1] == i][0].iloc[0] for i in top_indices]
    '''
    top_recommendations = []
    for idx in top_indices:
        if idx not in rated_model_indices:
            db_idx = mapping_table[mapping_table.iloc[1] == idx][0].values[0]
            top_recommendations.append(db_idx)
        if len(top_recommendations) >= top_k:
            break
    '''

    return recommendation

class BoardgameDV(DetailView):
    model = Boardgame_detail
    template_name = 'list/post_detail.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['disqus_short'] = f"{settings.DISQUS_SHORTNAME}"
        context['disqus_id'] = f"post-{self.object.id}"
        context['disqus_url'] = f"{settings.DISQUS_MY_DOMAIN}{self.object.get_absolute_url()}"
        context['disqus_title'] = f"{self.object.primary}"

        # Get recommendations
        content_recommendations = self.get_content_based_recommendation()
        #collaborative_recommendations = self.get_colaborative_filtering_recommendation(self.object.primary)
        collaborative_recommendations = self.get_ncf_recommendations(self.object.index)
        random_recommendation = self.get_random_recommendation()

        context['content_recommendations'] = Boardgame_detail.objects.filter(index__in=content_recommendations)
        context['collaborative_recommendations'] = Boardgame_detail.objects.filter(index__in=collaborative_recommendations)
        rating_choices = list(range(10, 0, -1))  # 10점부터 1점까지

        context['rating_choices'] = rating_choices


            # 유저가 이미 평점 남겼는지 확인
        if self.request.user.is_authenticated:
            try:
                existing_rating = Rating.objects.get(user=self.request.user, boardgame=self.object)
                context['user_rating'] = float(existing_rating.rating)
            except Rating.DoesNotExist:
                context['user_rating'] = None
            personalized_ids = self.get_personalized_recommendation(self.request.user)
            context['personalized_recommendations'] = Boardgame_detail.objects.filter(index__in=personalized_ids)
        else:
            context['user_rating'] = None
            context['personalized_recommendations'] = []

        return context


    def get_content_based_recommendation(self, k=10):
        # Get the index of the target game
        try:
            target_idx = self.object.index
        except ValueError:
            return []
        # Find similar games
        similar_idx = similarity[target_idx, :].argsort()[-(k+1):][::-1]
        similar_idx = similar_idx[similar_idx != target_idx]


        return similar_idx

    def get_colaborative_filtering_recommendation(self, board_game_title, k=10):
        # Get the index of the target game
        bg_titles_list = list(bg_titles)
        try:
            target_idx = bg_titles_list.index(board_game_title)
        except ValueError:
            return []

        # Retrieve the indices of the top-k similar games
        similar_idx = np.nonzero(top_k_corr_matrix[target_idx])[0]
        top_k_similar_idx = similar_idx[:k]
        recommendation = list(bg_titles[top_k_similar_idx])

        return recommendation

    def get_ncf_recommendations(self, db_index, k=10):
        # model.eval()

        target_idx = mapping_table[mapping_table.iloc[:,0] == db_index][1].iloc[0]

        similar_idx = ncf_corr_matrix[target_idx]
        top_k_indices = [item[0] for item in similar_idx[:k]]

        recommendation = [mapping_table[mapping_table.iloc[:,1] == i][0].iloc[0] for i in top_k_indices]

        return recommendation


    def get_personalized_recommendation(self, user, top_k=10):
        user_ratings = Rating.objects.filter(user=user)

        rating_list = [(r.boardgame.index, r.rating) for r in user_ratings]
        if not rating_list:
            return []

        recommended_ids = recommend_for_new_user(
            user_ratings=rating_list,
            top_k=top_k
        )
        return recommended_ids


    def get_random_recommendation(self, k=10):
        total_games = Boardgame_detail.objects.all().values_list('index', flat=True)
        random_indices = random.sample(list(total_games), k)

        return random_indices


def boardgame_search(request):
    form = BoardgameSearchForm(request.GET or None)
    boardgames = Boardgame_detail.objects.all()

    if form.is_valid():
        q = form.cleaned_data.get('q')
        min_players = form.cleaned_data.get('min_players')
        max_players = form.cleaned_data.get('max_players')
        min_playingtime = form.cleaned_data.get('min_playingtime')
        max_playingtime = form.cleaned_data.get('max_playingtime')
        min_average = form.cleaned_data.get('min_average')
        max_average = form.cleaned_data.get('max_average')

        if q:
            boardgames = boardgames.filter(
                Q(primary__icontains=q) | Q(alternate__icontains=q)
            )
        if min_players:
            boardgames = boardgames.filter(minplayers__gte=min_players)
        if max_players:
            boardgames = boardgames.filter(maxplayers__lte=max_players)
        if min_playingtime:
            boardgames = boardgames.filter(playingtime__gte=min_playingtime)
        if max_playingtime:
            boardgames = boardgames.filter(playingtime__lte=max_playingtime)
        if min_average:
            boardgames = boardgames.filter(average__gte=min_average)
        if max_average:
            boardgames = boardgames.filter(average__lte=max_average)

    # Pagination
    paginator = Paginator(boardgames, 12)  # 12 items per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Get the base query parameters without the 'page' parameter
    query_params = request.GET.copy()
    if 'page' in query_params:
        del query_params['page']

    return render(request, 'list/boardgame_search.html', {
        'form': form,
        'boardgames': page_obj,
        'query_params': query_params.urlencode()
    })

from django.shortcuts import get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .models import Boardgame_detail, Rating

@login_required
def rate_boardgame(request, boardgame_id):
    if request.method == 'POST':
        rating_value = float(request.POST.get('rating'))
        boardgame = get_object_or_404(Boardgame_detail, index=boardgame_id)

        # 기존 평가가 있으면 업데이트, 없으면 새로 생성
        rating, created = Rating.objects.update_or_create(
            user=request.user,
            boardgame=boardgame,
            defaults={'rating': rating_value}
        )
    return redirect(boardgame.get_absolute_url())
