from django.conf.urls import url
from django.views.decorators.csrf import csrf_exempt

from . import views

urlpatterns = [

    url(r'^home$', views.home, name='home'),
    url(r'^download$', views.download, name='download'),
]


