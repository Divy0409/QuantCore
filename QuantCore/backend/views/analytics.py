from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.generic import TemplateView
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

load_dotenv()
API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')

class StockDataPageView(TemplateView):
    template_name = 'analytics.html'

class StockDataAnalysisAPIView(APIView):
    def get(self, request):
        ticker = request.GET.get('ticker', '').upper()
        # ✅ Only run analytics if ticker provided
        if not ticker:
            return Response({"message": "Welcome to Stock Data Analysis API. Please provide a ticker to analyze."}, status=200)

        # ---------- 1️⃣ Fetch Full Historical Daily Data ----------
        price_url = (
            f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED'
            f'&symbol={ticker}&outputsize=full&apikey={API_KEY}'
        )
        price_res = requests.get(price_url)
        price_json = price_res.json()

        # Handle API note or missing data
        if "Note" in price_json:
            return Response({
                'error': 'Alpha Vantage API limit reached. Please wait a minute and try again.',
                'details': price_json["Note"]
            }, status=429)

        time_series_key = next((k for k in price_json if 'Time Series' in k), None)
        if not time_series_key:
            return Response({
                'error': 'Failed to fetch stock data or API limit reached.',
                'details': price_json.get("Note", "No Time Series data found.")
            }, status=400)

        time_series = price_json[time_series_key]
        historical_data = []
        for date, val in sorted(time_series.items()):
            historical_data.append({
                'date': date,
                'open': float(val['1. open']),
                'high': float(val['2. high']),
                'low': float(val['3. low']),
                'close': float(val['4. close']),
                'adjusted_close': float(val.get('5. adjusted close', val['4. close'])),
                'volume': int(val['6. volume'])
            })
        print(f"Fetched {len(historical_data)} days of historical data for {ticker}.")
        
        # ---------- 2️⃣ Fetch Sentiment Data ----------
        sentiment_url = (
            f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT'
            f'&tickers={ticker}&apikey={API_KEY}'
        )
        sentiment_json = requests.get(sentiment_url).json()

        sentiment_data = []
        if "feed" in sentiment_json:
            for item in sentiment_json["feed"]:
                time_published = item.get("time_published", "")
                title = item.get("title", "")
                url = item.get("url", "")
                if "ticker_sentiment" in item:
                    for ts in item["ticker_sentiment"]:
                        if ts.get("ticker") == ticker:
                            sentiment_data.append({
                                "time_published": time_published,
                                "title": title,
                                "url": url,
                                "relevance_score": ts.get("relevance_score"),
                                "sentiment_score": ts.get("ticker_sentiment_score"),
                                "sentiment_label": ts.get("ticker_sentiment_label")
                            })
        print(f"Fetched {len(sentiment_data)} sentiment articles for {ticker}.")

        # ---------- 3️⃣ Optional Company Info ----------
        overview_url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={API_KEY}'
        overview_res = requests.get(overview_url).json()
        company_name = overview_res.get("Name", ticker)

        # ---------- 4️⃣ Prepare DataFrames ----------
        df_prices = pd.DataFrame(historical_data)
        df_prices['date'] = pd.to_datetime(df_prices['date'])
        df_prices.sort_values('date', inplace=True)

        df_sent = pd.DataFrame(sentiment_data)
        if not df_sent.empty:
            df_sent['date'] = pd.to_datetime(df_sent['time_published']).dt.date
            
            # Convert to numeric safely
            df_sent['sentiment_score'] = pd.to_numeric(df_sent['sentiment_score'], errors='coerce')
            
            # Drop rows that failed conversion
            df_sent.dropna(subset=['sentiment_score'], inplace=True)
            
            # Now safely average by date
            df_sent = df_sent.groupby('date')['sentiment_score'].mean().reset_index()

            df_sent.rename(columns={'sentiment_score': 'daily_sentiment'}, inplace=True)
        else:
            # If no sentiment data, create placeholder
            df_sent = pd.DataFrame(columns=['date', 'daily_sentiment'])

        # Merge safely
        df_prices['date_only'] = df_prices['date'].dt.date
        df = pd.merge(df_prices, df_sent, left_on='date_only', right_on='date', how='left')

        # Drop only columns that exist
        for col in ['date', 'date_only']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        df['daily_sentiment'] = pd.to_numeric(df['daily_sentiment'], errors='coerce').fillna(0)

        # ---------- 5️⃣ Feature Engineering ----------
        if len(df) < 15:
            return Response({'error': 'Insufficient historical data returned by API.'}, status=400)

        df['return'] = df['close'].pct_change()
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['volatility_5'] = df['return'].rolling(window=5).std()
        df['sentiment_ma3'] = df['daily_sentiment'].rolling(window=3).mean()
        df['target'] = df['close'].shift(-1)
        df = df[df['target'].notna()]

        print(f"df shape after feature engineering: {df.shape}")
        print(df.head(5))

        # ---------- 🧠 Missing Value Analysis ----------
        def analyze_missing_values(df):
            missing_df = df.isnull().sum().to_frame('missing_count')
            missing_df['missing_percent'] = (missing_df['missing_count'] / len(df)) * 100
            missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(by='missing_percent', ascending=False)

            if missing_df.empty:
                print("\n✅ No missing values found in the dataset after feature engineering.")
            else:
                print("\n📊 Missing Values Analysis (After Feature Engineering):")
                for idx, row in missing_df.iterrows():
                    print(f" - {idx}: {row['missing_count']} missing ({row['missing_percent']:.2f}%)")

            return missing_df

        analyze_missing_values(df)

        # ---------- 6️⃣ Train-Test Split ----------
        features = ['open', 'high', 'low', 'close', 'volume',
                    'ma_5', 'ma_10', 'volatility_5',
                    'daily_sentiment', 'sentiment_ma3']

        X = df[features]
        y = df['target']

        if len(df) < 30:
            return Response({'error': 'Not enough data to train model.'}, status=400)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # ---------- 7️⃣ Train XGBoost Model ----------
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        # ---------- 8️⃣ Evaluate ----------
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        # ---------- 9️⃣ Predict Next-Day Price ----------
        last_row = df.iloc[-1:]
        next_day_pred = float(model.predict(last_row[features])[0])

        # ---------- ✅ Final Response ----------
        return Response({
            'ticker': ticker,
            'company_name': company_name,
            'rmse': rmse,
            'r2': r2,
            'predicted_next_day_price': next_day_pred,
            'historical_data_count': len(historical_data),
            'sentiment_data_count': len(sentiment_data)
        }, status=status.HTTP_200_OK)