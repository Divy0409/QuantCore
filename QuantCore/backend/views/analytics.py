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
        if not ticker:
            return Response({"message": "Welcome to Stock Data Analysis API. Please provide a ticker to analyze."}, status=200)

        # ---------- 1️⃣ Fetch Historical Data ----------
        interval = "5min"
        price_url = (
            f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY'
            f'&symbol={ticker}&interval={interval}&outputsize=full&apikey={API_KEY}'
        )
        price_json = requests.get(price_url).json()   
        # Handle API note or missing data
        if "Note" in price_json:
            return Response({
                'error': 'Alpha Vantage API limit reached. Please wait a minute and try again.',
                'details': price_json["Note"]
            }, status=429)

        time_series_key = next((k for k in price_json if 'Time Series' in k), None)
        if not time_series_key:
            return Response({'error': 'Failed to fetch stock data or API limit reached.','details': price_json.get("Note", "No Time Series data found.")}, status=400)

        time_series = price_json[time_series_key]
        historical_data = [
            {
                'date': pd.to_datetime(date),
                'open': float(v['1. open']),
                'high': float(v['2. high']),
                'low': float(v['3. low']),
                'close': float(v['4. close']),
                # 'adjusted_close': float(v.get('5. adjusted close', v['4. close'])),
                'volume': int(v['5. volume'])
            }
            for date, v in sorted(time_series.items())
        ]
        print(f"Fetched {len(historical_data)} days of historical data for {ticker}.")

        # ---------- 2️⃣ Fetch Sentiment Data ----------
        sentiment_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={API_KEY}&limit=1000'
        sentiment_json = requests.get(sentiment_url).json()

        sentiment_data = []
        target_ticker = ticker.upper()
        if "feed" in sentiment_json:
            for item in sentiment_json["feed"]:
                time_published = item.get("time_published", "")
                title = item.get("title", "")
                url = item.get("url", "")
                if "ticker_sentiment" in item:
                    for ts in item["ticker_sentiment"]:
                        if ts.get("ticker") == target_ticker:
                            sentiment_data.append({
                                "time_published": time_published,
                                "title": title,
                                "url": url,
                                "ticker": ts.get("ticker"),
                                "relevance_score": ts.get("relevance_score"),
                                "sentiment_score": ts.get("ticker_sentiment_score"),
                                "sentiment_label": ts.get("ticker_sentiment_label")
                            })

        if len(sentiment_data) == 0:
            return Response({'error': 'No sentiment data available.'}, status=200)

        print(f"Fetched {len(historical_data)} days of historical data for {ticker}.")                    
        print(f"Fetched {len(sentiment_data)} sentiment articles for {ticker}.")
        print(sentiment_data[:5])  # Print first 5 sentiment data entries for debugging
        print("sentiment_scores:", [d["sentiment_score"] for d in sentiment_data[:500000]])

        # ---------- 3️⃣ Optional Company Info ----------
        overview_url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={API_KEY}'
        overview_res = requests.get(overview_url).json()
        company_name = overview_res.get("Name", ticker)


        # ---------- 3️⃣ Convert to DataFrames ----------
        df_prices = pd.DataFrame(historical_data).sort_values("date")

        df_sent = pd.DataFrame(sentiment_data)
        if not df_sent.empty:
            # Convert timestamp and extract date correctly
            df_sent["date"] = pd.to_datetime(df_sent["time_published"], errors="coerce").dt.date
            df_prices["date"] = pd.to_datetime(df_prices["date"]).dt.date

            # Convert to numeric safely
            df_sent["sentiment_score"] = pd.to_numeric(df_sent["sentiment_score"], errors="coerce")
            df_sent.dropna(subset=["sentiment_score"], inplace=True)
        else:
            df_sent = pd.DataFrame(columns=["date", "daily_sentiment"])

        # ---------- 4️⃣ Merge ----------
        df = pd.merge(df_prices, df_sent, on="date", how="left")
        df.rename(columns={"sentiment_score": "daily_sentiment"}, inplace=True)

        # Replace NaN sentiment scores with 0
        df["daily_sentiment"] = pd.to_numeric(df["daily_sentiment"], errors="coerce").fillna(0)

        # ---------- 5️⃣ Feature Engineering ----------
        if len(df) < 15:
            return Response({'error': 'Insufficient historical data returned by API.'}, status=400)

        # Keep only direct day-to-day derived metrics — no averaging
        df["return"] = df["close"].pct_change()
        df["price_diff"] = df["close"].diff()  # optional: raw price change
        df["volatility"] = df["return"].abs()  # daily absolute change as volatility proxy
        df["target"] = df["close"].shift(-1)
        df = df.dropna(subset=["target"])

        drop_cols = [
            "time_published", "title", "url", 
            "ticker", "relevance_score", "sentiment_label"
        ]

        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        print(f"df shape after feature engineering: {df.shape}")
        print(df.head())

        # ---------- 🧠 Missing Values ----------
        def analyze_missing_values(df, consider_zero_as_missing=True):
            missing_mask = df.isnull()
            
            if consider_zero_as_missing:
                # Add boolean mask for zeros
                zero_mask = df.select_dtypes(include=[np.number]) == 0
                missing_mask = missing_mask | zero_mask

            missing_count = missing_mask.sum().to_frame('missing_count')
            missing_count['missing_percent'] = (missing_count['missing_count'] / len(df)) * 100
            missing_count = missing_count[missing_count['missing_count'] > 0].sort_values(by='missing_percent', ascending=False)

            if missing_count.empty:
                print("\n✅ No missing values found in the dataset after feature engineering.")
            else:
                print("\n📊 Missing Values Analysis (After Feature Engineering):")
                for idx, row in missing_count.iterrows():
                    print(f" - {idx}: {row['missing_count']} missing ({row['missing_percent']:.2f}%)")
            
            return missing_count

        analyze_missing_values(df)

        # ---------- 6️⃣ Train Model ----------
        features = ["open", "high", "low", "close", "volume", "price_diff", "volatility", "daily_sentiment"]
        X, y = df[features], df["target"]

        if len(df) < 30:
            return Response({'error': 'Not enough data to train model.'}, status=400)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        # ---------- 7️⃣ Predict Next Day ----------
        next_day_pred = float(model.predict(df[features].iloc[[-1]])[0])

        # ---------- ✅ Final Response ----------
        return Response({
            "ticker": ticker,
            "company_name": company_name,
            "rmse": rmse,
            "r2": r2,
            "predicted_next_day_price": next_day_pred,
            "historical_data_count": len(historical_data),
            "sentiment_data_count": len(sentiment_data)
        }, status=200)
