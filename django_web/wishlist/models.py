from django.db import models
from django.contrib.auth.models import User

# Create your models here.
# This describes how our saved data looks like

class Wishlist(models.Model):
    title=models.CharField('TITLE',max_length=100,blank=True)
    url = models.URLField('URL',unique=True)
    owner=models.ForeignKey(User,on_delete=models.CASCADE,verbose_name="OWNER",blank=True,null=True)

    def __str__(self):
        return self.title
