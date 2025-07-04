from django.contrib import admin
from .models.chatbot import Conversation as ConversationTable

admin.site.register(ConversationTable)