from tracemalloc import start

import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_curve, auc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.preprocessing import StandardScaler 
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# ============================================================
#   LSTM MODEL
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ============================================================
#   TRAINING HELPERS
# ============================================================

def train_model(model, X_train, y_train, epochs=20, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=32, shuffle=True
    )

    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
    
def fetch_indicator(function_name, ticker, interval="daily", time_period=None):
    base_url = "https://www.alphavantage.co/query"

    params = {
        "function": function_name,
        "symbol": ticker,
        "interval": interval,
        "apikey": API_KEY
    }

    if time_period:
        params["time_period"] = time_period
        params["series_type"] = "close"

    if function_name == "MACDEXT":
        params["series_type"] = "close"

    response = requests.get(base_url, params=params).json()

    key = next((k for k in response if "Technical Analysis" in k), None)
    if not key:
        return pd.DataFrame()

    records = []
    for date_str, values in response[key].items():
        row = {"date": pd.to_datetime(date_str)}
        for col, val in values.items():
            row[col.lower()] = float(val)
        records.append(row)

    df_indicator = pd.DataFrame(records)
    return df_indicator.sort_values("date")

# ============================================================
#   MAIN FUNCTION
# ============================================================

def NeuralTechnicalModel(ticker: str, user_date: str = None):

    price_url = (
        f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY'
        f'&symbol={ticker}&outputsize=full&apikey={API_KEY}'
    )
    time.sleep(12)  # To respect API rate limits

    price_json = requests.get(price_url).json()

    if "Note" in price_json:
        return {"error": "API limit reached"}

    ts_key = next((k for k in price_json if "Time Series" in k), None)
    if not ts_key:
        return {"error": "No time series returned"}

    df = pd.DataFrame([
        {
            "date": pd.to_datetime(d),
            "open": float(v["1. open"]),
            "high": float(v["2. high"]),
            "low": float(v["3. low"]),
            "close": float(v["4. close"]),
            "volume": int(v["5. volume"])
        }
        for d, v in sorted(price_json[ts_key].items())
    ]).sort_values("date").reset_index(drop=True)

    if len(df) < 200:
        return {"error": "Insufficient data"}

    # ====================================================
    # Technical Features
    # ====================================================

    df["return_1"] = df["close"].pct_change()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()

    # ====================================================
    # Pull Technical Indicators from Alpha Vantage
    # ====================================================

    df_rsi = fetch_indicator("RSI", ticker, interval="daily", time_period=14)
    time.sleep(12)

    df_mom = fetch_indicator("MOM", ticker, interval="daily", time_period=10)
    time.sleep(12)

    df_macd = fetch_indicator("MACDEXT", ticker, interval="daily")
    time.sleep(12)

    # Merge with main dataframe
    for indicator_df in [df_rsi, df_mom, df_macd]:
        if not indicator_df.empty:
            df = pd.merge(df, indicator_df, on="date", how="left")

    actual_last_date = df["date"].iloc[-1]
    df["return"] = df["close"].pct_change()
    df["target_return"] = df["return"].shift(-1)
    df = df.dropna()

    # ====================================================
    # Calculate Days Ahead (if user provided date)
    # ====================================================

    if user_date:
        future_date = datetime.strptime(user_date, "%Y-%m-%d")

        def trading_days_between(start, end):
             return np.busday_count(start.date(), end.date())
        
        days_ahead = trading_days_between(actual_last_date, future_date)

        if days_ahead <= 0:
            return {
                "error": "Provided date must be in the future relative to the latest data point."
            }
    else:
        days_ahead = 1  # default to next day

    features = ["close", "open", "high", "low", "volume", "return_1", "ma_5", "ma_10", "rsi", "mom", "macd", "macd_signal", "macd_hist"]

    df = df.dropna().reset_index(drop=True)

    # ====================================================
    # Create Sequences
    # ====================================================

    seq = 15
    X_seq = []
    y_seq = []

    for i in range(len(df) - seq):
        X_seq.append(df[features].iloc[i:i+seq].values)
        y_seq.append(df["target_return"].iloc[i+seq])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # ====================================================
    # Scaling (NO LEAKAGE)
    # ====================================================

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Reshape 3D -> 2D for scaling (samples * seq, features)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    # Fit ONLY on training data
    feature_scaler.fit(X_train_reshaped)

    # Transform train + test
    X_train = feature_scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_test = feature_scaler.transform(X_test_reshaped).reshape(X_test.shape)

    # Scale target (fit only on train)
    target_scaler.fit(y_train.reshape(-1, 1))

    y_train = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_test = target_scaler.transform(y_test.reshape(-1, 1)).flatten()


    # ====================================================
    # Train LSTM
    # ====================================================

    model = LSTMModel(input_size=len(features))
    train_model(model, X_train, y_train, epochs=20)


    # ====================================================
    # Predict & Inverse Transform
    # ====================================================

    # Predict daily returns for test set
    y_pred_test_scaled = predict(model, X_test)

    # Convert back to real returns (inverse scaling)
    y_pred_test = target_scaler.inverse_transform(
        y_pred_test_scaled.reshape(-1, 1)
    ).flatten()

    y_test = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # ====================================================
    # FINAL METRICS (Post-Prediction Evaluation Layer)
    # ====================================================

    # Residuals
    residuals = y_test - y_pred_test

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2 = float(r2_score(y_test, y_pred_test))
    mae = float(mean_absolute_error(y_test, y_pred_test))

    # ---------------------------
    # Theil’s U
    # ---------------------------
    def theils_u(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        naive = y_true[:-1]   # naive forecast = previous actual
        actual = y_true[1:]
        forecast = y_pred[1:]

        num = np.sqrt(np.mean((forecast - actual) ** 2))
        den = np.sqrt(np.mean((actual - naive) ** 2))

        return float(num / den)

    theils_u_value = theils_u(y_test, y_pred_test)

    # ====================================================
    # Directional Metrics (Aligned Properly)
    # ====================================================

    actual_direction = (y_test > 0).astype(int)
    predicted_direction = (y_pred_test > 0).astype(int)

    directional_accuracy = float(
        np.mean(actual_direction == predicted_direction)
    )

    # ====================================================
    # ROC Curve (Uses Continuous Scores)
    # ====================================================

    # Continuous score = predicted price change
    pred_scores = y_pred_test  # continuous score = predicted return

    if len(np.unique(actual_direction)) > 1:
        roc_fpr, roc_tpr, _ = roc_curve(actual_direction, pred_scores)
        roc_auc = float(auc(roc_fpr, roc_tpr))

        roc_fpr = roc_fpr.tolist()
        roc_tpr = roc_tpr.tolist()
    else:
        roc_fpr, roc_tpr, roc_auc = None, None, None

    # ====================================================
    # Predict Next Price
    # ====================================================
    # ====================================================
    # Forecast Future Price (Recursive)
    # ====================================================

    def forecast_future_price(
        model,
        feature_scaler,
        target_scaler,
        df,
        features,
        seq,
        days_ahead
    ):
        last_close = df["close"].iloc[-1]

        current_seq_raw = df[features].iloc[-seq:].values
        current_seq_scaled = feature_scaler.transform(
            current_seq_raw
        ).reshape(1, seq, -1)

        predicted_returns = []

        for _ in range(days_ahead):

            next_scaled = predict(model, current_seq_scaled)

            next_return = target_scaler.inverse_transform(
                next_scaled.reshape(-1, 1)
            )[0][0]

            predicted_returns.append(next_return)

            # Shift sequence forward (simple persistence)
            next_feature_row = current_seq_scaled[:, -1:, :]
            current_seq_scaled = np.concatenate(
                [current_seq_scaled[:, 1:, :], next_feature_row],
                axis=1
            )

        expected_price = last_close
        for r in predicted_returns:
            expected_price *= (1 + r)

        return expected_price, predicted_returns

    # CALL FORECAST
    expected_price, predicted_returns = forecast_future_price(
        model,
        feature_scaler,
        target_scaler,
        df,
        features,
        seq,
        days_ahead
    )

    residual_std = np.std(residuals)  # from test set
    def compute_confidence_interval(expected_price, residual_std, days_ahead, confidence=0.95):

            z = 1.96  # 95%

            # variance grows with horizon
            total_std = residual_std * np.sqrt(days_ahead)

            lower = expected_price - z * total_std * expected_price
            upper = expected_price + z * total_std * expected_price

            return float(lower), float(upper)


    ci_lower, ci_upper = compute_confidence_interval(
        expected_price,
        residual_std,
        days_ahead
    )

    latest_seq_raw = df[features].iloc[-seq:].values

    latest_seq_scaled = feature_scaler.transform(
        latest_seq_raw
    ).reshape(1, seq, -1)

    next_scaled = predict(model, latest_seq_scaled)

    next_return = target_scaler.inverse_transform(
        next_scaled.reshape(-1,1)
    )[0][0]

    last_close = df["close"].iloc[-1]
    next_price_prediction = float(last_close * (1 + next_return))

    overview_url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={API_KEY}'
    overview_res = requests.get(overview_url).json()
    company_name = overview_res.get("Name", ticker)

    return {
    "ticker": ticker,
    "company_name": company_name,
    "predicted_next_close": next_price_prediction,
    "long_term_forecast": {
        "days_ahead": days_ahead,
        "expected_price": expected_price,
        "confidence_interval": [ci_lower, ci_upper],
        "predicted_returns": predicted_returns
    },
    "rmse": rmse,
    "r2": r2,
    "mae": mae,
    "theils_u": theils_u_value,
    "directional_accuracy": directional_accuracy,
    "roc_curve": {
        "fpr": roc_fpr,
        "tpr": roc_tpr,
        "auc": roc_auc
    },
    "price_history_points": len(df),
    "mean_residual": float(np.mean(residuals)),
    "residual_std": float(np.std(residuals))
}