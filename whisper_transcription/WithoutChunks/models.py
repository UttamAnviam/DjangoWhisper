from django.db import models

class AudioFile(models.Model):
    audio_name = models.CharField(max_length=255)
    uuid = models.CharField(max_length=255, unique=True)
    status = models.CharField(max_length=50, choices=[('pending', 'Pending'), ('processing', 'Processing'), ('completed', 'Completed'), ('error', 'Error'), ('timeout', 'Timeout')])
    created_at = models.DateTimeField(auto_now_add=True)
