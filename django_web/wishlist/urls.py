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
from django.urls import path
#from wishlist.views import WishlistLV,WishlistDV
from wishlist import views

# Below shows how URLs in wishlist are routed to views

app_name='wishlist'
urlpatterns = [
    path('',views.WishlistLV.as_view(),name='index'),

    path('<int:pk>/',views.WishlistDV.as_view(),name='detail'),

    path('add/',views.WishlistCreateView.as_view(),name="add",),

    path('change/',views.WishlistChangeLV.as_view(),name="change",),

    path('<int:pk>/update/',views.WishlistUpdateView.as_view(),name="update",),

    path('<int:pk>/delete/',views.WishlistDeleteView.as_view(),name="delete",),
]
