from django.urls import path
from .views import NoChunksTranscribeView

urlpatterns = [
    path('noChunks_transcribe/', NoChunksTranscribeView.as_view(), name='transcribe_chunks'),
#   path('health/', health_check, name='health_check'),
]
