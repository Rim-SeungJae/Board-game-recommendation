"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from list import views
from django.urls import path,re_path
from .views import boardgame_search

app_name='list'

# Below shows how URLs related to postes are routed to views

urlpatterns = [

    path('',views.BoardgameLV.as_view(),name='index'),

    path('post/',views.BoardgameDV.as_view(),name='post_list'),

    path('post/<int:pk>/', views.BoardgameDV.as_view(),name='post_detail'),

    path('search/',views.boardgame_search,name='boardgame_search'),
]
