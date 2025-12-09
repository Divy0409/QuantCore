import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

load_dotenv()
API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')

def XGBoost(ticker: str):
    # ---------- 1️⃣ Fetch Historical Data ----------
        interval = "5min"
        price_url = (
            f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY'
            f'&symbol={ticker}&interval={interval}&outputsize=full&apikey={API_KEY}'
        )
        price_json = requests.get(price_url).json()   
        # Handle API note or missing data
        if "Note" in price_json:
            return {
                'error': 'Alpha Vantage API limit reached. Please wait a minute and try again.',
                'details': price_json["Note"]
            }

        time_series_key = next((k for k in price_json if 'Time Series' in k), None)
        if not time_series_key:
            return {
                'error': 'Failed to fetch stock data or API limit reached.',
                'details': price_json.get("Note", "No Time Series data found.")
            }
        time_series = price_json[time_series_key]
        historical_data = [
            {
                'date': pd.to_datetime(d),
                'open': float(v['1. open']),
                'high': float(v['2. high']),
                'low': float(v['3. low']),
                'close': float(v['4. close']),
                'adjusted_close': float(v.get('5. adjusted close', v['4. close'])),
                'volume': int(v['5. volume'])
            }
            for d, v in sorted(time_series.items())
        ]
        print(f"Fetched {len(historical_data)} days of historical data for {ticker}.")

        df_daily = pd.DataFrame(historical_data).sort_values("date").reset_index(drop=True)
        if df_daily.empty or len(df_daily) < 90:
            return ({'error': 'Insufficient daily historical data.'}, 400)

        # ---------- 2) Fetch sentiment data ----------
        sent_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={API_KEY}&limit=1000"
        sent_json = requests.get(sent_url).json()
        if "Note" in sent_json:
            # sentiment API limited => proceed with baseline-only
            sentiment_available = False
            df_sent_daily = pd.DataFrame()
        else:
            sentiment_data = []
            target_ticker = ticker.upper()
            if "feed" in sent_json:
                for item in sent_json["feed"]:
                    if "ticker_sentiment" not in item:
                        continue
                    for tsent in item["ticker_sentiment"]:
                        if tsent.get("ticker") == target_ticker:
                            # keep time_published and sentiment score
                            sentiment_data.append({
                                "time_published": item.get("time_published"),
                                "title": item.get("title"),
                                "url": item.get("url"),
                                "ticker": tsent.get("ticker"),
                                "relevance_score": tsent.get("relevance_score"),
                                "sentiment_score": tsent.get("ticker_sentiment_score"),
                                "sentiment_label": tsent.get("ticker_sentiment_label"),
                            })
            df_sent = pd.DataFrame(sentiment_data)
            if df_sent.empty:
                sentiment_available = False
                df_sent_daily = pd.DataFrame()
            else:
                sentiment_available = True
                # convert timestamp and aggregate to daily sentiment features
                df_sent["datetime"] = pd.to_datetime(df_sent["time_published"], errors="coerce")
                df_sent["date"] = df_sent["datetime"].dt.date
                df_sent["sentiment_score"] = pd.to_numeric(df_sent["sentiment_score"], errors="coerce")
                df_sent.dropna(subset=["sentiment_score"], inplace=True)

                # aggregate per-date: mean, count, std
                agg = df_sent.groupby("date")["sentiment_score"].agg(["mean", "count", "std"]).reset_index()
                agg.rename(columns={"mean": "sent_mean", "count": "sent_count", "std": "sent_std"}, inplace=True)
                # convert agg.date back to datetime aligned to df_daily
                agg["date"] = pd.to_datetime(agg["date"])
                df_sent_daily = agg.copy()

        # ---------- 3) Prepare baseline dataset (Model A) ----------
        df = df_daily.copy()
        df["date_only"] = df["date"].dt.date  # helper

        # Feature engineering for baseline: lags, returns, rolling stats
        df["return_1"] = df["adjusted_close"].pct_change(1)
        df["return_2"] = df["adjusted_close"].pct_change(2)
        df["ma_5"] = df["adjusted_close"].rolling(window=5).mean()
        df["ma_10"] = df["adjusted_close"].rolling(window=10).mean()
        df["vol_5"] = df["return_1"].rolling(window=5).std()
        df["vol_10"] = df["return_1"].rolling(window=10).std()
        df["target_next_close"] = df["adjusted_close"].shift(-1)  # predicting next day adjusted_close

        df_baseline = df.dropna(subset=["target_next_close", "return_1", "ma_5", "ma_10"]).reset_index(drop=True)

        baseline_features = ["adjusted_close", "open", "high", "low", "volume", "return_1", "return_2", "ma_5", "ma_10", "vol_5", "vol_10"]

        # safety check
        if len(df_baseline) < 200:
            # still try but warn user
            pass

        # Train/Test split (no shuffle — time series)
        train_size = int(len(df_baseline) * 0.8)
        X_base = df_baseline[baseline_features]
        y_base = df_baseline["target_next_close"]

        X_base_train, X_base_test = X_base.iloc[:train_size], X_base.iloc[train_size:]
        y_base_train, y_base_test = y_base.iloc[:train_size], y_base.iloc[train_size:]

        model_base = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model_base.fit(X_base_train, y_base_train)

        # Evaluate baseline (informational)
        try:
            yb_pred = model_base.predict(X_base_test)
            base_rmse = float(np.sqrt(mean_squared_error(y_base_test, yb_pred)))
            base_r2 = float(r2_score(y_base_test, yb_pred))
        except Exception:
            base_rmse, base_r2 = None, None

        # ---------- 4) If sentiment available: prepare data for Model B ----------
        model_b = None
        sentiment_rmse = None
        sentiment_r2 = None

        if sentiment_available and not df_sent_daily.empty:
            # merge daily baseline features with sentiment daily features on date
            # Align df_baseline dates to calendar date for merge
            df_baseline_for_sent = df_baseline.copy()
            df_baseline_for_sent["date_only"] = df_baseline_for_sent["date"].dt.date
            # baseline prediction for next close on the day row (i.e., predict next day using features at day t)
            baseline_preds = model_base.predict(df_baseline_for_sent[baseline_features])
            df_baseline_for_sent["baseline_pred_next"] = baseline_preds  # baseline prediction of next day's close

            # build DataFrame with actual next day close aligned with current row (already target_next_close)
            # Now merge sentiment daily aggregates by date (we want sentiment that occurred on day t to explain residual for day t->t+1)
            # Ensure df_sent_daily date align to midnight
            df_sent_daily["date_only"] = pd.to_datetime(df_sent_daily["date"]).dt.date

            merged = pd.merge(
                df_baseline_for_sent,
                df_sent_daily[["date_only", "sent_mean", "sent_count", "sent_std"]],
                left_on="date_only",
                right_on="date_only",
                how="inner"  # only keep days where sentiment exists
            ).reset_index(drop=True)

            if merged.empty or len(merged) < 30:
                # Not enough overlap to train sentiment model reliably
                sentiment_available = False
            else:
                # residual target = actual_next_close - baseline_pred_next
                merged["residual_next"] = merged["target_next_close"] - merged["baseline_pred_next"]

                # Features for sentiment model: sent_mean, sent_count, sent_std, short recent price moves
                # Add short-term features: recent return and vol
                merged["recent_return_1"] = merged["return_1"]
                merged["recent_return_2"] = merged["return_2"]
                merged["recent_vol_5"] = merged["vol_5"]
                sent_features = ["sent_mean", "sent_count", "sent_std", "recent_return_1", "recent_return_2", "recent_vol_5"]

                # Drop rows with NaN in any feature (especially sent_std can be NaN)
                merged = merged.dropna(subset=sent_features + ["residual_next"]).reset_index(drop=True)

                if len(merged) < 30:
                    sentiment_available = False
                else:
                    # Train sentiment model
                    X_sent = merged[sent_features]
                    y_sent = merged["residual_next"]

                    # train/test split
                    train_n = int(len(X_sent) * 0.8)
                    Xs_train, Xs_test = X_sent.iloc[:train_n], X_sent.iloc[train_n:]
                    ys_train, ys_test = y_sent.iloc[:train_n], y_sent.iloc[train_n:]

                    model_b = xgb.XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42
                    )
                    model_b.fit(Xs_train, ys_train)
                    ys_pred = model_b.predict(Xs_test)
                    sentiment_rmse = float(np.sqrt(mean_squared_error(ys_test, ys_pred)))
                    sentiment_r2 = float(r2_score(ys_test, ys_pred))

    # ---------- 5) Evaluate combined (hybrid) model on historical test set ----------
        combined_rmse = None
        combined_r2 = None

        # Prepare df_baseline_test to align dates with X_base_test / y_base_test
        df_baseline_test = df_baseline.iloc[train_size:].reset_index(drop=True)

        if len(X_base_test) > 0:
            combined_preds = []
            combined_actuals = []

            # create lookup for daily aggregated sentiment by date if available
            if sentiment_available and not df_sent_daily.empty:
                lookup_sent = df_sent_daily.copy()
                lookup_sent["date_only"] = pd.to_datetime(lookup_sent["date"]).dt.date
                lookup_sent = lookup_sent.set_index("date_only")
            else:
                lookup_sent = None

            for i in range(len(X_base_test)):
                # corresponding row in df_baseline_test
                row = df_baseline_test.iloc[i]
                test_actual = float(y_base_test.iloc[i])

                # baseline prediction for this test row (predict next close using features at t)
                b_pred = float(model_base.predict(X_base_test.iloc[[i]])[0])

                # sentiment prediction for this date (if available)
                s_pred = 0.0
                if lookup_sent is not None and model_b is not None:
                    test_date = row["date_only"]  # a datetime.date
                    if test_date in lookup_sent.index:
                        sent_row = lookup_sent.loc[test_date]
                        sent_mean = float(sent_row["sent_mean"])
                        sent_count = float(sent_row["sent_count"])
                        sent_std = float(sent_row["sent_std"]) if not np.isnan(sent_row["sent_std"]) else 0.0

                        # recent price features from the baseline row
                        recent_return_1 = float(row["return_1"]) if not np.isnan(row["return_1"]) else 0.0
                        recent_return_2 = float(row["return_2"]) if not np.isnan(row["return_2"]) else 0.0
                        recent_vol_5 = float(row["vol_5"]) if not np.isnan(row["vol_5"]) else 0.0

                        sent_feat_row = pd.DataFrame([{
                            "sent_mean": sent_mean,
                            "sent_count": sent_count,
                            "sent_std": sent_std,
                            "recent_return_1": recent_return_1,
                            "recent_return_2": recent_return_2,
                            "recent_vol_5": recent_vol_5
                        }])[["sent_mean", "sent_count", "sent_std", "recent_return_1", "recent_return_2", "recent_vol_5"]]

                        try:
                            s_pred = float(model_b.predict(sent_feat_row)[0])
                        except Exception:
                            s_pred = 0.0

                # combined prediction and actual
                combined_pred_val = b_pred + s_pred
                combined_preds.append(combined_pred_val)
                combined_actuals.append(test_actual)

            # compute metrics if we have any predictions
            if len(combined_preds) > 0:
                combined_rmse = float(np.sqrt(mean_squared_error(combined_actuals, combined_preds)))
                combined_r2 = float(r2_score(combined_actuals, combined_preds))

        # ---------- 6) Make predictions for the next day ----------
        # Prepare the most recent row of df_daily to compute baseline prediction for next day
        latest_row = df_daily.iloc[-1:].copy()
        # build baseline features for latest row similar to df_baseline
        latest_row["return_1"] = df_daily["adjusted_close"].pct_change(1).iloc[-1]
        latest_row["return_2"] = df_daily["adjusted_close"].pct_change(2).iloc[-1]
        latest_row["ma_5"] = df_daily["adjusted_close"].rolling(window=5).mean().iloc[-1]
        latest_row["ma_10"] = df_daily["adjusted_close"].rolling(window=10).mean().iloc[-1]
        latest_row["vol_5"] = df_daily["adjusted_close"].pct_change().rolling(window=5).std().iloc[-1]
        latest_row["vol_10"] = df_daily["adjusted_close"].pct_change().rolling(window=10).std().iloc[-1]

        # if any baseline feature is NaN (e.g., very start), we cannot predict
        if latest_row[baseline_features].isnull().any(axis=None):
            return {'error': 'Not enough history in daily data to compute baseline features.'}

        baseline_pred = float(model_base.predict(latest_row[baseline_features])[0])

        sentiment_effect = 0.0
        # If sentiment model is available, compute today's sentiment aggregated and predict residual effect
        if sentiment_available and model_b is not None:
            # We try to extract today's sentiment (matching the latest date)
            # Note: df_sent_daily has aggregated sentiment per date (date = day sentiment occurred)
            last_date = df_daily["date"].dt.date.iloc[-1]  # latest trading day
            # There may be sentiment entries for that date or for the last calendar day (news published after close may occur)
            # We'll look for sentiment matching last_date (or the most recent available)
            df_sent_daily["date_only"] = pd.to_datetime(df_sent_daily["date"]).dt.date
            row_sent = df_sent_daily[df_sent_daily["date_only"] == last_date]
            if row_sent.empty:
                # fallback: use most recent sentiment daily aggregate available
                row_sent = df_sent_daily.sort_values("date_only", ascending=False).iloc[[0]]

            # construct features for model_b
            sent_feat_row = {
                "sent_mean": float(row_sent["sent_mean"].iloc[0]),
                "sent_count": float(row_sent["sent_count"].iloc[0]),
                "sent_std": float(row_sent["sent_std"].iloc[0]) if not np.isnan(row_sent["sent_std"].iloc[0]) else 0.0,
                # recent price features from latest_row
                "recent_return_1": float(latest_row["return_1"].iloc[0]) if not np.isnan(latest_row["return_1"].iloc[0]) else 0.0,
                "recent_return_2": float(latest_row["return_2"].iloc[0]) if not np.isnan(latest_row["return_2"].iloc[0]) else 0.0,
                "recent_vol_5": float(latest_row["vol_5"].iloc[0]) if not np.isnan(latest_row["vol_5"].iloc[0]) else 0.0,
            }

            Xb = pd.DataFrame([sent_feat_row])[["sent_mean", "sent_count", "sent_std", "recent_return_1", "recent_return_2", "recent_vol_5"]]
            try:
                sentiment_effect = float(model_b.predict(Xb)[0])
            except Exception:
                sentiment_effect = 0.0

        # Final prediction
        final_pred = baseline_pred + sentiment_effect
    
        # ---------- 6) Optional Company Info ----------
        overview_url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={API_KEY}'
        overview_res = requests.get(overview_url).json()
        company_name = overview_res.get("Name", ticker)

        # ---------- 7) Construct response ----------
        return {
                    "ticker": ticker,
                    "company_name": company_name,
                    "baseline_pred_next_close": baseline_pred,
                    "sentiment_effect_pred": sentiment_effect,
                    "final_pred_next_close": final_pred,
                    "baseline_rmse": base_rmse, 
                    "baseline_r2": base_r2,
                    "sentiment_rmse": sentiment_rmse,
                    "sentiment_r2": sentiment_r2,
                    "final_rmse": combined_rmse,
                    "final_r2": combined_r2,
                    "price_history_days": len(df_daily),
                    "sentiment_days_available": int(df_sent_daily.shape[0]) if sentiment_available else 0,
                }