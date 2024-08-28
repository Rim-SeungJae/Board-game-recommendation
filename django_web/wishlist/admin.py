from django.contrib import admin
from wishlist.models import Wishlist

# Register models here.

@admin.register(Wishlist)
class WishlistAdmin(admin.ModelAdmin):
    list_display=('id','title','url')
