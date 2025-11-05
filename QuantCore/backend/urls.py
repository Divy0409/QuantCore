from django.urls import path
from .views.portfolio import PortfolioAPIView
from .views.stock_views import StockDataAPIView, stock_chart_page
from .views.analytics import StockDataAnalysisAPIView, StockDataPageView
from .views.settings import settings_view
from .views.chatbot import chatbot_view
from .views.home import home_view
from .views import SessionCreateView, ConversationListView, ConversationCreateView, SessionListView, ChatbotGenerateResponseView
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path('chatbot/', SessionCreateView.as_view(), name='create-session'),
    path('chatbot/sessions/', SessionListView.as_view(), name='list-sessions'),
    path('stock-data/', StockDataAPIView.as_view(), name='stock-data-api'),
    path('stock/', stock_chart_page, name='stock-page'),
    path('analytics/', StockDataPageView.as_view(), name='analytics-view'),
    path('analytics-data/', StockDataAnalysisAPIView.as_view(), name='analytics-data'),
    path('chatbot/<str:session_id>/', ConversationListView.as_view(), name='get-conversation'),
    path('chatbot/<str:session_id>/add/', ConversationCreateView.as_view(), name='post-message'),
    path('chatbot/<str:session_id>/response/', ChatbotGenerateResponseView.as_view(), name='generate-response'),
    path('chatbot_view/', chatbot_view, name='chatbot-view'),  
    path('home_view/', home_view, name='home-view'),  
    path('settings_view/', settings_view, name='settings-view'),
    path('portfolio', PortfolioAPIView.as_view(), name='portfolio-view'),

]


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)