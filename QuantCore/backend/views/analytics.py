from rest_framework.views import APIView
from rest_framework.response import Response
from django.views.generic import TemplateView
from ..Analytics_Model_Engines.XGBoost import XGBoost 
from ..Analytics_Model_Engines.NeuralHybrid import NeuralHybrid
from ..Analytics_Model_Engines.Advanced import advance_analysis
from ..Analytics_Model_Engines.PureNeural import NeuralTechnicalModel
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
        user_date = request.GET.get('date', None)  # Optional date for analysis (e.g., "2024-01-01")

        if not ticker:
            return Response({
                "message": "Welcome to Stock Data Analysis API. Please provide a ticker to analyze.",
                "available_models": ["xgboost", "neural", "advanced"]
            }, status=200)
        
        # Select model based on user input
        if model_choice == 'xgboost':
            model = XGBoost
        elif model_choice == 'hybridneural':
            model = NeuralHybrid
        elif model_choice == 'pureneural':
            model = NeuralTechnicalModel

            # Validate date format ONLY if provided
            if user_date:
                try:
                    from datetime import datetime
                    datetime.strptime(user_date, "%Y-%m-%d")
                except ValueError:
                    return Response({
                        "error": "Invalid date format. Use YYYY-MM-DD"
                    }, status=400)

            result = model(ticker, user_date=user_date)
        else:
            return Response({
                "message": f"Invalid model choice '{model_choice}'.",
                "available_models": ["xgboost", "hybridneural", "pureneural"]
            }, status=400)

        # Run the chosen model
        result = model(ticker,user_date=user_date) if model_choice == 'pureneural' else model(ticker)
        return Response(result, status=200)