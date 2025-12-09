import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# ============================================================
#   NEURAL NETWORK MODELS
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]       # last timestep
        return self.fc(out)


class SentimentMLP(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


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


# ============================================================
#   MAIN FUNCTION
# ============================================================

def NeuralHybrid(ticker: str):
    # ====================================================
    # 1) FETCH PRICE DATA
    # ====================================================
    interval = "5min"
    price_url = (
        f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY'
        f'&symbol={ticker}&interval={interval}&outputsize=full&apikey={API_KEY}'
    )
    price_json = requests.get(price_url).json()

    if "Note" in price_json:
        return {"error": "API limit reached", "details": price_json["Note"]}

    ts_key = next((k for k in price_json if "Time Series" in k), None)
    if not ts_key:
        return {"error": "No time series returned"}

    df_daily = pd.DataFrame([ { "date": pd.to_datetime(d), "open": float(v["1. open"]), "high": float(v["2. high"]), "low": float(v["3. low"]), "close": float(v["4. close"]), "adjusted_close": float(v.get("5. adjusted close", v["4. close"])), "volume": int(v["5. volume"]) } for d, v in sorted(price_json[ts_key].items()) ]).sort_values("date").reset_index(drop=True)

    if len(df_daily) < 200:
        return {"error": "Insufficient historical data"}

    # ------------------------------------------------------
    # Baseline features
    # ------------------------------------------------------
    df_daily["return_1"] = df_daily["adjusted_close"].pct_change(1) 
    df_daily["return_2"] = df_daily["adjusted_close"].pct_change(2)
    df_daily["ma_5"] = df_daily["adjusted_close"].rolling(5).mean() 
    df_daily["ma_10"] = df_daily["adjusted_close"].rolling(10).mean() 
    df_daily["vol_5"] = df_daily["return_1"].rolling(5).std() 
    df_daily["vol_10"] = df_daily["return_1"].rolling(10).std() 
    df_daily["target_next_close"] = df_daily["adjusted_close"].shift(-1) 

    baseline_features = [
        "adjusted_close", "open", "high", "low", "volume",
        "return_1", "return_2", "ma_5", "ma_10", "vol_5", "vol_10"
    ]

    df_base = df_daily.dropna(subset=baseline_features + ["target_next_close"]).reset_index(drop=True)

    # ------------------------------------------------------
    # Create sequences for LSTM: 15-day windows
    # ------------------------------------------------------
    seq = 15
    X_seq = []
    y_seq = []

    for i in range(len(df_base) - seq):
        X_seq.append(df_base[baseline_features].iloc[i:i+seq].values)
        y_seq.append(df_base["target_next_close"].iloc[i+seq])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # ====================================================
    # 2) TRAIN BASELINE LSTM
    # ====================================================
    lstm = LSTMModel(input_size=len(baseline_features))
    train_model(lstm, X_train, y_train, epochs=15)

    base_pred_test = predict(lstm, X_test)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, base_pred_test)))
    baseline_r2 = float(r2_score(y_test, base_pred_test))

    # ====================================================
    # 3) FETCH SENTIMENT & TRAIN SENTIMENT MODEL
    # ====================================================
    sent_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={API_KEY}&limit=1000"
    sent_json = requests.get(sent_url).json()

    sentiment_available = False
    df_sent_daily = pd.DataFrame()

    if "feed" in sent_json:
        sentiment_available = True
        sentiment_data = []
        for item in sent_json["feed"]:
            if "ticker_sentiment" not in item:
                continue
            for tsent in item["ticker_sentiment"]:
                if tsent.get("ticker") == ticker.upper():
                    sentiment_data.append({ "datetime": pd.to_datetime(item["time_published"], errors="coerce"), "sentiment_score": float(tsent.get("ticker_sentiment_score", 0)) })

        df_sent = pd.DataFrame(sentiment_data)
        if df_sent.empty:
            sentiment_available = False
        else: 
            df_sent["date_only"] = df_sent["datetime"].dt.date 
            agg = df_sent.groupby("date_only")["sentiment_score"].agg(["mean","std","count"]).reset_index() 
            df_sent_daily = agg.copy()

    model_b = None 
    sentiment_rmse = None 
    sentiment_r2 = None 
    combined_rmse = None 
    combined_r2 = None 

    # Compute baseline predictions for df_base
    df_base["baseline_pred_next"] = np.nan
    for i in range(seq, len(df_base)):
        seq_x = df_base[baseline_features].iloc[i-seq:i].values.reshape(1, seq, -1)
        df_base.loc[i, "baseline_pred_next"] = float(predict(lstm, seq_x))

    if sentiment_available and not df_sent_daily.empty:
        # Merge sentiment with baseline
        df_base["date_only"] = df_base["date"].dt.date
        merged = pd.merge(df_base, df_sent_daily, on="date_only", how="inner")
        merged["residual_next"] = merged["target_next_close"] - merged["baseline_pred_next"]
        
        # Use features: mean, std, count + recent returns/vol
        merged["recent_return_1"] = merged["return_1"]
        merged["recent_return_2"] = merged["return_2"]
        merged["recent_vol_5"] = merged["vol_5"]
        sent_features = ["mean","count","std","recent_return_1","recent_return_2","recent_vol_5"]
        merged = merged.dropna(subset=sent_features + ["residual_next"])
        
        if len(merged) >= 30:
            Xs = merged[sent_features].values
            ys = merged["residual_next"].values
            split_s = int(0.8 * len(Xs))
            Xs_train, Xs_test = Xs[:split_s], Xs[split_s:]
            ys_train, ys_test = ys[:split_s], ys[split_s:]
            model_b = SentimentMLP(input_dim=len(sent_features))
            train_model(model_b, Xs_train, ys_train, epochs=20)
            ys_pred = predict(model_b, Xs_test)
            sentiment_rmse = float(np.sqrt(mean_squared_error(ys_test, ys_pred)))
            sentiment_r2 = float(r2_score(ys_test, ys_pred))
        
            # Combined predictions
            combined_pred = merged["baseline_pred_next"].values + predict(model_b, merged[sent_features].values)
            combined_rmse = float(np.sqrt(mean_squared_error(merged["target_next_close"], combined_pred)))
            combined_r2 = float(r2_score(merged["target_next_close"], combined_pred))
    
    # ----------  Predict next day ----------
    latest_seq = df_base[baseline_features].iloc[-seq:].values.reshape(1, seq, -1)
    baseline_pred_next = float(predict(lstm, latest_seq))
    sentiment_effect_next = 0.0
    if sentiment_available and model_b is not None:
        last_date = df_base["date"].iloc[-1].date()
        row_sent = df_sent_daily[df_sent_daily["date_only"] == last_date]
        if row_sent.empty:
            row_sent = df_sent_daily.sort_values("date_only", ascending=False).iloc[[0]]
        X_sent_pred = row_sent[["mean","count","std"]].copy()
        X_sent_pred["recent_return_1"] = df_base["return_1"].iloc[-1]
        X_sent_pred["recent_return_2"] = df_base["return_2"].iloc[-1]
        X_sent_pred["recent_vol_5"] = df_base["vol_5"].iloc[-1]
        X_sent_pred = X_sent_pred[["mean","count","std","recent_return_1","recent_return_2","recent_vol_5"]].values
        sentiment_effect_next = float(predict(model_b, X_sent_pred))
    final_pred_next = baseline_pred_next + sentiment_effect_next

    overview_url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={API_KEY}'
    overview_res = requests.get(overview_url).json()
    company_name = overview_res.get("Name", ticker)

    return {
        "ticker": ticker,
        "company_name": company_name,
        "baseline_pred_next_close": baseline_pred_next,
        "sentiment_effect_pred": sentiment_effect_next,
        "final_pred_next_close": final_pred_next,
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2,
        "sentiment_rmse": sentiment_rmse,
        "sentiment_r2": sentiment_r2,
        "final_rmse": combined_rmse,
        "final_r2": combined_r2,
        "price_history_days": len(df_daily),
        "sentiment_days_available": int(df_sent_daily.shape[0]) if sentiment_available else 0
    }