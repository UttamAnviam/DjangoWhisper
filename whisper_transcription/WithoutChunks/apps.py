import asyncio
from django.apps import AppConfig
import logging

class WithoutChunksConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "WithoutChunks"

 
