# skincare_ai/urls.py  (PROJECT-LEVEL — not home/urls.py)

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('home.urls')),   # ← ALL home app URLs at root
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)