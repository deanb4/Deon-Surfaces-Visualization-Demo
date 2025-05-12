from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_image, name = 'upload_image'),
    # path('render/', views.render_realistic, name = 'render_realistic'),
    path('', views.home, name = 'home_page')
]