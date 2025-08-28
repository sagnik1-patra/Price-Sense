import os
from celery import Celery
from .utils import read_settings

settings = read_settings()
app = Celery("pricesense", broker=settings.get("redis_url", "redis://localhost:6379/0"))
app.conf.timezone = "UTC"
