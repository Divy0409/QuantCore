from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
import os
from dotenv import load_dotenv
from django.shortcuts import render

def stock_chart_page(request):
    return render(request, 'stock_view.html')

load_dotenv()
API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')

class StockDataAPIView(APIView):
    def get(self, request):
        ticker = request.GET.get('ticker', '').upper()
        interval = request.GET.get('interval', 'daily')

        if not ticker:
            return Response({'error': 'Ticker is required.'}, status=400)

        function_map = {
            'daily': 'TIME_SERIES_DAILY',
            'weekly': 'TIME_SERIES_WEEKLY',
            'monthly': 'TIME_SERIES_MONTHLY'
        }

        function = function_map.get(interval, 'TIME_SERIES_DAILY')
        url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={API_KEY}'
        response = requests.get(url)
        data = response.json()

        time_series_key = next((k for k in data if 'Time Series' in k), None)

        if not time_series_key:
            return Response({'error': 'Failed to fetch data or limit reached.', 'details': data.get("Note")}, status=400)

        time_series = data[time_series_key]
        chart_data = sorted(
            [(date, float(val['4. close'])) for date, val in time_series.items()],
            key=lambda x: x[0]
        )

        # Limit to last 5 years (roughly 252 trading days per year)
        if interval == '5y':
            chart_data = chart_data[-(5 * 252):]
        elif interval == 'max':
            chart_data = chart_data

        dates = [item[0] for item in chart_data]
        prices = [item[1] for item in chart_data]
        company_name = ticker

        overview_url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={API_KEY}'
        overview_res = requests.get(overview_url).json()

        if "Name" in overview_res:
            company_name = overview_res["Name"]
        else:
            search_url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={ticker}&apikey={API_KEY}'
            search_res = requests.get(search_url).json()
            if "bestMatches" in search_res and len(search_res["bestMatches"]) > 0:
                company_name = search_res["bestMatches"][0]["2. name",ticker]

        return Response({
            'ticker': ticker,
            'company_name': company_name,
            'interval': interval,
            'dates': dates,
            'prices': prices
        })
