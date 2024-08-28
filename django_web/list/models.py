from django.db import models
from django.urls import reverse
from django.contrib.auth.models import User
from django.utils.text import slugify

class Boardgame(models.Model):
    index = models.IntegerField(verbose_name='index',primary_key=True)
    id = models.IntegerField(verbose_name='id')
    thumbnail = models.TextField(verbose_name='thumbnail')
    image = models.TextField(verbose_name='image')
    primary = models.TextField(verbose_name='primary')

    class Meta:
        verbose_name='boardgame'
        verbose_name_plural='boardgames'
        db_table='bg_list'
        ordering=('index',)

    def __str__(self):
        return self.primary

    def get_absolute_url(self):
        return reverse('list:post_detail',args=(self.index,))

class Boardgame_detail(models.Model):
    index = models.IntegerField(verbose_name='index',primary_key=True)
    id = models.IntegerField(verbose_name='id')
    thumbnail = models.TextField(verbose_name='thumbnail')
    image = models.TextField(verbose_name='image')
    primary = models.TextField(verbose_name='primary')
    description = models.TextField(verbose_name='description')
    minplayers = models.IntegerField()
    maxplayers = models.IntegerField()
    playingtime = models.IntegerField()
    board_game_rank = models.IntegerField()
    boardgamecategory = models.TextField()
    average = models.FloatField()
    alternate = models.TextField()

    class Meta:
        verbose_name='boardgame_detail'
        verbose_name_plural='boardgame_details'
        db_table='bg_info'

    def __str__(self):
        return self.primary

    def get_absolute_url(self):
        return reverse('list:post_detail',args=(self.index,))
