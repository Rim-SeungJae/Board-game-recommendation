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
        managed = False

    def __str__(self):
        return self.primary

    def get_absolute_url(self):
        return reverse('list:post_detail',args=(self.index,))

class Boardgame_detail(models.Model):
    index = models.IntegerField(verbose_name='index',primary_key=True)
    id = models.IntegerField(verbose_name='id')
    thumbnail = models.TextField(default = '', verbose_name='thumbnail')
    image = models.TextField(verbose_name='image')
    primary = models.TextField(verbose_name='primary')
    description = models.TextField(verbose_name='description')
    minplayers = models.IntegerField()
    maxplayers = models.IntegerField()
    playingtime = models.IntegerField()
    board_game_rank = models.IntegerField(default = '')
    boardgamecategory = models.TextField()
    average = models.FloatField()
    alternate = models.TextField(default = '')

    class Meta:
        verbose_name='boardgame_detail'
        verbose_name_plural='boardgame_details'
        db_table='bg_info'
        managed = False

    def __str__(self):
        return self.primary

    def get_absolute_url(self):
        return reverse('list:post_detail',args=(self.index,))

class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    boardgame = models.ForeignKey('list.Boardgame_detail', on_delete=models.CASCADE)
    rating = models.FloatField()
    rated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'boardgame')  # 중복 평가 방지
        managed = False

    def __str__(self):
        return f'{self.user.username} rated {self.boardgame.primary} - {self.rating}'
