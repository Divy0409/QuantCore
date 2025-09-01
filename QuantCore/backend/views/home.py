from django.shortcuts import render
from django.conf import settings
from django.conf.urls.static import static

from huggingface_hub import HfApi
from decouple import config

DEFAULT_HF_KEY = config("HF_API_KEY", default="")
DEFAULT_HF_ACCOUNT = config("HF_ACCOUNT_NAME", default=None)

def home_view(request):
    return render(request, 'home.html')