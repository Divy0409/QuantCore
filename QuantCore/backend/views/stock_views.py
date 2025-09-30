from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
import os
from dotenv import load_dotenv
from django.shortcuts import render
from datetime import datetime,timedelta

def stock_chart_page(request):
    return render(request, 'stock_view.html')

load_dotenv()
API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')
FMP_API_KEY = os.getenv('FMP_API_KEY')

class StockDataAPIView(APIView):
    def get(self, request):
        ticker = request.GET.get('ticker', '').upper()
        interval = request.GET.get('interval', 'daily')

        if not ticker:
            return Response({'error': 'Ticker is required.'}, status=400)

        # ---------- Chart Data ----------
        # Handle Alpha Vantage functions
        if interval in ['1min', '5min', '15min', '30min', '60min']:
            function = "TIME_SERIES_INTRADAY"
            url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&interval={interval}&apikey={API_KEY}'
        elif interval == 'daily':
            function = "TIME_SERIES_INTRADAY"
            intraday_interval = '60min'
            url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&interval={intraday_interval}&outputsize=compact&apikey={API_KEY}'
        elif interval in ['weekly','monthly','5y']:
            function = "TIME_SERIES_DAILY"
            url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&outputsize=full&apikey={API_KEY}'
        elif interval == 'yearly':
            function = "TIME_SERIES_MONTHLY"
            url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={API_KEY}'
        else:
            # Default to daily
            function = "TIME_SERIES_DAILY"
            url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={API_KEY}'
       
        response = requests.get(url)
        data = response.json()

        time_series_key = next((k for k in data if 'Time Series' in k), None)

        if not time_series_key:
            return Response({'error': 'Failed to fetch data or limit reached.', 'details': data.get("Note")}, status=400)

        time_series = data[time_series_key]
        sorted_data = sorted(time_series.items(), key=lambda x: x[0])

        now = datetime.now()
        chart_data = []

        if interval == 'daily':
           # All intraday timestamps
            intraday_points = [(dt, float(val['4. close'])) for dt, val in sorted_data]
            
            # Find the most recent date (YYYY-MM-DD) in the data
            latest_date = max(dt.split(" ")[0] for dt, _ in intraday_points)
            
            # Keep only data for that date
            chart_data = [(dt, price) for dt, price in intraday_points if dt.startswith(latest_date)]
            
        elif interval == 'weekly':
            year, week, _ = now.isocalendar()
            chart_data = []
            for dt, val in sorted_data:
                d = datetime.strptime(dt, '%Y-%m-%d')
                y, w, _ = d.isocalendar()
                if y == year and w == week:
                    chart_data.append((dt, float(val['4. close'])))

        elif interval == 'monthly':
            this_month = now.strftime('%Y-%m')
            chart_data = [(dt, float(val['4. close'])) for dt, val in sorted_data if dt.startswith(this_month)]

        elif interval == 'yearly':
            this_year = now.year
            for dt, val in sorted_data:
                d = datetime.strptime(dt, '%Y-%m-%d')
                if d.year == this_year:
                    chart_data.append((dt, float(val['4. close'])))

        elif interval == '5y':
            five_years_ago = now - timedelta(days=5*365)
            for dt, val in sorted_data:
                d = datetime.strptime(dt, '%Y-%m-%d')
                if d >= five_years_ago:
                    chart_data.append((dt, float(val['4. close'])))

        else:
            chart_data = [(dt, float(val['4. close'])) for dt, val in sorted_data]

        dates = [item[0] for item in chart_data]
        prices = [item[1] for item in chart_data]
        company_name = ticker

        # ---------- Company Name ----------
        overview_url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={API_KEY}'
        overview_res = requests.get(overview_url).json()

        if "Name" in overview_res:
            company_name = overview_res["Name"]
        else:
            search_url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={ticker}&apikey={API_KEY}'
            search_res = requests.get(search_url).json()
            if "bestMatches" in search_res and len(search_res["bestMatches"]) > 0:
                company_name = search_res["bestMatches"][0]["2. name",ticker]

        # ---------- Sentiment Data ----------
        sentiment_url = (f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={API_KEY}'
)
        sentiment_json = requests.get(sentiment_url).json()

        sentiment_data = []
        target_ticker = ticker.upper()
        if "feed" in sentiment_json:
            for item in sentiment_json["feed"]:
                source = item.get("source", "").lower(),
                time_published = item.get("time_published", "")
                if "ticker_sentiment" in item:
                    for ts in item["ticker_sentiment"]:
                        if ts.get("ticker") == target_ticker:
                            sentiment_data.append({
                                "source": source,
                                "time_published": time_published,
                                "ticker": ts.get("ticker"),
                                "relevance_score": ts.get("relevance_score"),
                                "ticker_sentiment_score": ts.get("ticker_sentiment_score"),
                                "ticker_sentiment_label": ts.get("ticker_sentiment_label"),
                        })

        return Response({
            'ticker': ticker,
            'company_name': company_name,
            'interval': interval,
            'dates': dates,
            'prices': prices,
            'sentiment': sentiment_data
        })
