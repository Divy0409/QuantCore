from rest_framework.views import APIView
from rest_framework.response import Response
from django.views.generic import TemplateView
from ..Analytics_Model_Engines.XGBoost import XGBoost 
from ..Analytics_Model_Engines.NeuralHybrid import NeuralHybrid
from ..Analytics_Model_Engines.Advanced import advance_analysis
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')

class StockDataPageView(TemplateView):
    template_name = 'analytics.html'

class StockDataAnalysisAPIView(APIView):
    """
    Two-model approach:
      - Model A (baseline): trained on daily historical prices (no sentiment).
      - Model B (sentiment residual): trained on overlap period where sentiment exists.
    Response includes baseline_pred, sentiment_effect, final_prediction.
    """
    def get(self, request):
        ticker = request.GET.get('ticker', '').upper()
        model_choice = request.GET.get('model', 'neural').lower()  # default to NeuralHybrid

        if not ticker:
            return Response({
                "message": "Welcome to Stock Data Analysis API. Please provide a ticker to analyze.",
                "available_models": ["xgboost", "neural", "advanced"]
            }, status=200)
        
        # Select model based on user input
        if model_choice == 'xgboost':
            model = XGBoost
        elif model_choice == 'neural':
            model = NeuralHybrid
        elif model_choice == 'advanced':
            model = advance_analysis
        else:
            return Response({
                "message": f"Invalid model choice '{model_choice}'.",
                "available_models": ["xgboost", "neural", "advanced"]
            }, status=400)

        # Run the chosen model
        result = model(ticker)
        return Response(result, status=200)