from django.views.generic import ListView, DetailView
from django.views.generic.dates import ArchiveIndexView, YearArchiveView,MonthArchiveView
from django.views.generic.dates import DayArchiveView, TodayArchiveView

from list.models import Boardgame, Boardgame_detail
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

model = NCF(351048,18984,20,[64,32])
model_load_path = os.path.join(settings.MEDIA_ROOT, 'models', 'ncf_model.pth')
model.load_state_dict(torch.load(model_load_path,map_location=torch.device('cpu')))

class BoardgameLV(ListView):
    model=Boardgame
    template_name='list/post_all.html'
    context_object_name='boardgames'
    paginate_by=18

# Load the npy files
similarity_path = os.path.join(settings.MEDIA_ROOT, 'npys', 'similarity.npy')
corr_matrix_path = os.path.join(settings.MEDIA_ROOT, 'npys', 'corr_matrix.npy')
ncf_corr_matrix_path = os.path.join(settings.MEDIA_ROOT, 'npys', 'ncf_corr_matrix.npy');
bg_titles_path = os.path.join(settings.MEDIA_ROOT, 'npys', 'bg_titles.npy')

bg_titles = np.load(bg_titles_path)
similarity = np.load(similarity_path)
corr_matrix = np.load(corr_matrix_path)
corr_matrix = corr_matrix.argsort()[:, ::-1]
ncf_corr_matrix = np.load(ncf_corr_matrix_path)

mapping_table_path = os.path.join(settings.MEDIA_ROOT, 'npys', 'map.npy')
mapping_table = np.load(mapping_table_path, allow_pickle=True)
mapping_table = pd.DataFrame(mapping_table)

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

        context['content_recommendations'] = Boardgame_detail.objects.filter(index__in=content_recommendations)
        context['collaborative_recommendations'] = Boardgame_detail.objects.filter(index__in=collaborative_recommendations)

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

        # Find similar games

        similar_idx = corr_matrix[target_idx, :k+1].reshape(-1)
        similar_idx = similar_idx[similar_idx != target_idx]

        recommendation = list(bg_titles[similar_idx])

        return recommendation

    def get_ncf_recommendations(self, db_index, k=10):
        model.eval()

        target_idx = mapping_table[mapping_table.iloc[:,0] == db_index][1].iloc[0]

        similar_idx = np.argsort(-ncf_corr_matrix[target_idx])[:k+1]
        print(similar_idx)
        similar_idx = similar_idx[similar_idx != target_idx]

        recommendation = [mapping_table[mapping_table.iloc[:,1] == i][0].iloc[0] for i in similar_idx]

        return recommendation


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
