from django.urls import path
from . import views

urlpatterns = [
    # ROOT → Welcome page (opens at 127.0.0.1:8000/)
    path('',        views.welcome,     name='welcome'),

    # Auth
    path('login/',  views.login_view,  name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),

    # Main app (protected)
    path('home/',   views.home,        name='home'),
    path('upload/', views.upload,      name='upload'),
    path('privacy/', views.privacy, name='privacy'),
]