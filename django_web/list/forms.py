from django import forms
from .models import Boardgame_detail

class BoardgameSearchForm(forms.Form):
    q = forms.CharField(required=False, label='Search')
    min_players = forms.IntegerField(required=False, label='Min Players')
    max_players = forms.IntegerField(required=False, label='Max Players')
    min_playingtime = forms.IntegerField(required=False, label='Min Playing Time')
    max_playingtime = forms.IntegerField(required=False, label='Max Playing Time')
    min_average = forms.FloatField(required=False, label='Min Average Rating')
    max_average = forms.FloatField(required=False, label='Max Average Rating')
    boardgamecategory = forms.CharField(required=False, label='Category')
