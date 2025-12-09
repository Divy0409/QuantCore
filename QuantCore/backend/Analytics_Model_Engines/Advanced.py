"""
Hybrid forecasting pipeline using multiple price and sentiment models.:
 - Price models: LSTM, LightGBM, CNN-LSTM, RandomForest
 - Sentiment models: FinBERT (seq classification -> numeric), RoBERTa (seq classification -> numeric),
   Embedding-based features -> XGBoost regression
 - Ensembles and final weighted prediction:
     price_ensemble = weighted average (inverse-RMSE) of 4 price models
     sentiment_ensemble = average of 3 sentiment effects (FinBERT, RoBERTa, Embedding-XGB)
     final_pred = w_price * price_ensemble + w_sent * sentiment_ensemble
"""
import os
import math
import time
import requests
import warnings
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ML libs
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

# Deep learning (Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam

# Transformers & embeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Torch for running transformers
import torch

warnings.filterwarnings("ignore")
load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

# Device for transformers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seeds for reproducibility (uncomment if you want deterministic-ish runs)
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
torch.manual_seed(seed)

# -------------------------- Utilities -------------------------------------

def fetch_intraday_or_daily(ticker: str, interval: str = "60min", use_intraday: bool = False) -> pd.DataFrame:
    """Fetch price data (daily adjusted recommended)."""
    if not API_KEY:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY in environment or .env")

    if use_intraday:
        url = (
            f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
            f"&symbol={ticker}&interval={interval}&outputsize=full&apikey={API_KEY}"
        )
    else:
        url = (
            f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
            f"&symbol={ticker}&outputsize=full&apikey={API_KEY}"
        )

    res = requests.get(url, timeout=30)
    data = res.json()
    if "Note" in data:
        raise RuntimeError("AlphaVantage API limit reached: " + data["Note"])
    time_series_key = next((k for k in data.keys() if "Time Series" in k), None)
    if time_series_key is None:
        raise RuntimeError("No time series data found: " + str(data))
    ts = data[time_series_key]
    rows = []
    for d, v in sorted(ts.items()):
        open_ = v.get("1. open") or v.get("1. Open")
        high_ = v.get("2. high") or v.get("2. High")
        low_ = v.get("3. low") or v.get("3. Low")
        close_ = v.get("4. close") or v.get("4. Close")
        adj_close_ = v.get("5. adjusted close", close_)
        vol_ = v.get("6. volume", v.get("5. volume", 0))
        rows.append({
            "date": pd.to_datetime(d),
            "open": float(open_),
            "high": float(high_),
            "low": float(low_),
            "close": float(close_),
            "adjusted_close": float(adj_close_),
            "volume": int(float(vol_)) if vol_ is not None else 0
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

def fetch_sentiment_feed(ticker: str) -> pd.DataFrame:
    """Fetch AlphaVantage NEWS_SENTIMENT feed and return article-level DataFrame (may be empty)."""
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={API_KEY}&limit=1000"
    res = requests.get(url, timeout=30)
    data = res.json()
    if "Note" in data:
        return pd.DataFrame()
    feed = data.get("feed", [])
    rows = []
    for item in feed:
        time_published = item.get("time_published", "")
        title = item.get("title", "")
        url_item = item.get("url", "")
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker") == ticker.upper():
                rows.append({
                    "time_published": time_published,
                    "title": title,
                    "url": url_item,
                    "ticker": ts.get("ticker"),
                    "relevance_score": ts.get("relevance_score"),
                    "sentiment_score": ts.get("ticker_sentiment_score") or ts.get("sentiment_score"),
                    "sentiment_label": ts.get("ticker_sentiment_label") or ts.get("sentiment_label")
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["time_published"], errors="coerce")
        df["date"] = df["datetime"].dt.date
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
        df = df.dropna(subset=["sentiment_score"])
    return df

# -------------------------- price feature engineering ----------------------

def prepare_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_1"] = df["adjusted_close"].pct_change(1)
    df["return_2"] = df["adjusted_close"].pct_change(2)
    df["ma_5"] = df["adjusted_close"].rolling(5).mean()
    df["ma_10"] = df["adjusted_close"].rolling(10).mean()
    df["vol_5"] = df["return_1"].rolling(5).std()
    df["vol_10"] = df["return_1"].rolling(10).std()
    df["target_next"] = df["adjusted_close"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df

# -------------------------- Ensure engineered price columns -----------------

def ensure_price_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure minimal engineered price columns exist on df:
      - return_1, return_2, ma_5, ma_10, vol_5, vol_10, target_next
      - keeps original order/dtypes where possible
    This is safe to run on raw fetched df or on already-prepared df.
    """
    df = df.copy()
    # ensure date sorted
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    # adjusted_close must exist for calculations
    if "adjusted_close" not in df.columns:
        # if only 'close' exists, copy it
        if "close" in df.columns:
            df["adjusted_close"] = df["close"]
        else:
            # nothing to compute; create zeros to avoid KeyErrors downstream
            df["adjusted_close"] = 0.0

    # compute returns
    if "return_1" not in df.columns:
        df["return_1"] = df["adjusted_close"].pct_change(1).fillna(0.0)
    if "return_2" not in df.columns:
        df["return_2"] = df["adjusted_close"].pct_change(2).fillna(0.0)

    # moving averages
    if "ma_5" not in df.columns:
        df["ma_5"] = df["adjusted_close"].rolling(5).mean().fillna(method="bfill").fillna(df["adjusted_close"])
    if "ma_10" not in df.columns:
        df["ma_10"] = df["adjusted_close"].rolling(10).mean().fillna(method="bfill").fillna(df["adjusted_close"])

    # volatilities
    if "vol_5" not in df.columns:
        df["vol_5"] = df["return_1"].rolling(5).std().fillna(0.0)
    if "vol_10" not in df.columns:
        df["vol_10"] = df["return_1"].rolling(10).std().fillna(0.0)

    # next-day target
    if "target_next" not in df.columns:
        df["target_next"] = df["adjusted_close"].shift(-1)

    # helper date_only
    if "date_only" not in df.columns and "date" in df.columns:
        df["date_only"] = pd.to_datetime(df["date"]).dt.date

    return df

# -------------------------- NN builds & sequence helpers -------------------

def build_lstm(input_shape, lr=1e-3):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr), loss="mse")
    return model

def build_cnn_lstm(input_shape, lr=1e-3):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(LSTM(48))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr), loss="mse")
    return model

def create_sequences(X: np.ndarray, y: np.ndarray, window=30) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# -------------------------- transformers helpers ----------------------------

def load_text_classifier(model_name: str):
    """Load tokenizer + AutoModelForSequenceClassification to DEVICE and return them."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def _classifier_scores_batch(texts: list, tokenizer, model, batch_size: int = 8):
    """
    Run classifier in batches on DEVICE and return numeric scores per text.
    Mapping:
      - If logits dim == 1 -> logits (regression style)
      - If 2 classes -> p_pos - p_neg
      - If 3+ classes -> weighted sum into [-1..1] with linear weights
    """
    import math
    scores = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**encoded)
            logits = out.logits.detach().cpu().numpy()
        # logits -> probs
        probs = softmax(logits, axis=1)
        n_classes = probs.shape[1]
        if n_classes == 1:
            batch_scores = logits.flatten().tolist()
        elif n_classes == 2:
            batch_scores = (probs[:, 1] - probs[:, 0]).tolist()
        else:
            weights = np.linspace(-1, 1, n_classes)
            batch_scores = (probs * weights).sum(axis=1).tolist()
        scores.extend(batch_scores)
    return np.array(scores)

def sentiment_score_from_classifier(texts: list, tokenizer, model, batch_size=8) -> np.ndarray:
    """Wrapper: returns numeric sentiment scores for texts using tokenizer+model safely."""
    if tokenizer is None or model is None or len(texts) == 0:
        return np.zeros(len(texts))
    # For low VRAM reduce batch_size
    if DEVICE.type == "cuda":
        batch_size = min(batch_size, 4)
    return _classifier_scores_batch(texts, tokenizer, model, batch_size=batch_size)

def softmax(x, axis=1):
    x = np.asarray(x)
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def load_sentence_transformer(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

def free_torch():
    try:
        torch.cuda.empty_cache()
    except:
        pass
# -------------------------- XGBoost trainer --------------------------------

def train_xgboost_regressor(X, y):
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, max_depth=5, learning_rate=0.05)
    model.fit(X, y)
    return model

# -------------------------- Price model training ---------------------------

def train_price_models(df_price: pd.DataFrame, seq_window: int = 30, epochs_lstm: int = 10, epochs_cnnlstm: int = 8, verbose=0):
    """
    Train LSTM, CNN-LSTM, LightGBM, RandomForest.
    Returns: price_models dict, holdout dict, metrics dict, prepared_df, features list
    """
    df = prepare_price_features(df_price)
    features = ["adjusted_close", "open", "high", "low", "volume", "return_1", "return_2", "ma_5", "ma_10", "vol_5", "vol_10"]
    X = df[features].values
    y = df["target_next"].values

    # sequences for sequential models
    X_seq, y_seq = create_sequences(X, y, window=seq_window)
    if len(X_seq) < 40:
        raise RuntimeError("Not enough data to train LSTM/CNN-LSTM sequences. Need more history.")

    train_n = int(0.8 * len(X_seq))
    Xs_train, Xs_test = X_seq[:train_n], X_seq[train_n:]
    ys_train, ys_test = y_seq[:train_n], y_seq[train_n:]

    # LSTM
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    lstm = build_lstm(input_shape=(Xs_train.shape[1], Xs_train.shape[2]))
    if epochs_lstm > 0:
        lstm.fit(Xs_train, ys_train, epochs=epochs_lstm, batch_size=32, verbose=verbose, callbacks=[callback])
    lstm_pred = lstm.predict(Xs_test).flatten()
    lstm_rmse = math.sqrt(mean_squared_error(ys_test, lstm_pred))
    lstm_r2 = r2_score(ys_test, lstm_pred)

    # CNN-LSTM
    callback_cnn = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    cnn_lstm = build_cnn_lstm(input_shape=(Xs_train.shape[1], Xs_train.shape[2]))
    if epochs_cnnlstm > 0:
        cnn_lstm.fit(Xs_train, ys_train, epochs=epochs_cnnlstm, batch_size=32, verbose=verbose, callbacks=[callback_cnn])
    cnn_pred = cnn_lstm.predict(Xs_test).flatten()
    cnn_rmse = math.sqrt(mean_squared_error(ys_test, cnn_pred))
    cnn_r2 = r2_score(ys_test, cnn_pred)

    # Tabular train/test
    X_tab = df[features].values
    y_tab = df["target_next"].values
    t_split = int(0.8 * len(X_tab))
    X_tab_train, X_tab_test = X_tab[:t_split], X_tab[t_split:]
    y_tab_train, y_tab_test = y_tab[:t_split], y_tab[t_split:]
    X_tab_train = X_tab_train.astype(np.float32)
    X_tab_test  = X_tab_test.astype(np.float32)
    y_tab_train = y_tab_train.astype(np.float32)
    y_tab_test  = y_tab_test.astype(np.float32)

    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1)
    lgb_model.fit(X_tab_train, y_tab_train)
    lgb_pred = lgb_model.predict(X_tab_test)
    lgb_rmse = math.sqrt(mean_squared_error(y_tab_test, lgb_pred))
    lgb_r2 = r2_score(y_tab_test, lgb_pred)

    # RandomForest
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tab_train, y_tab_train)
    rf_pred = rf.predict(X_tab_test)
    rf_rmse = math.sqrt(mean_squared_error(y_tab_test, rf_pred))
    rf_r2 = r2_score(y_tab_test, rf_pred)

    price_models = {"lstm": lstm, "cnn_lstm": cnn_lstm, "lightgbm": lgb_model, "random_forest": rf}
    holdout = {
        "seq_test": Xs_test, "seq_y_test": ys_test,
        "tab_test": X_tab_test, "tab_y_test": y_tab_test,
        "lstm_pred": lstm_pred, "cnn_pred": cnn_pred, "lgb_pred": lgb_pred, "rf_pred": rf_pred
    }
    metrics = {
        "lstm": {"rmse": lstm_rmse, "r2": lstm_r2},
        "cnn_lstm": {"rmse": cnn_rmse, "r2": cnn_r2},
        "lightgbm": {"rmse": lgb_rmse, "r2": lgb_r2},
        "random_forest": {"rmse": rf_rmse, "r2": rf_r2}
    }
    return price_models, holdout, metrics, df, features

# -------------------------- Sentiment pipeline (fixed RoBERTa) ---------------

def build_sentiment_ensemble(df_price: pd.DataFrame,
                             df_sent_articles: pd.DataFrame,
                             finbert_model_name: str = "yiyanghkust/finbert-tone",
                             roberta_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
                             embed_model_name: str = "all-MiniLM-L6-v2",
                             verbose=0):
    """
    Safe sequential loading: FinBERT -> RoBERTa -> embedder.
    Returns df_sent_features, availability dict, None (placeholder)
    """
    if df_sent_articles is None or df_sent_articles.empty:
        return None, {"finbert": False, "roberta": False, "embedder": False}, {"message": "No sentiment articles available."}

    df_sent = df_sent_articles.copy()
    if "time_published" in df_sent.columns:
        df_sent["datetime"] = pd.to_datetime(df_sent["time_published"], errors="coerce")
        df_sent["date"] = df_sent["datetime"].dt.date
    else:
        df_sent["date"] = pd.to_datetime(df_sent["date"]).dt.date

    # find text
    text_col = "title"
    if text_col not in df_sent.columns:
        candidates = [c for c in df_sent.columns if c.lower() in ("title", "text", "content", "summary")]
        if candidates:
            text_col = candidates[0]
        else:
            raise RuntimeError("No text column found in sentiment feed")

    grouped = df_sent.groupby("date")
    per_day_texts = {d: g[text_col].astype(str).tolist() for d, g in grouped}

    # FinBERT scoring
    finbert_ok = False
    finbert_means = {}
    try:
        if verbose: print("Loading FinBERT:", finbert_model_name, "on", DEVICE)
        tk_fin, mdl_fin = load_text_classifier(finbert_model_name)
        finbert_ok = True
        for d, texts in per_day_texts.items():
            if not texts:
                finbert_means[d] = 0.0
                continue
            try:
                scores = sentiment_score_from_classifier(texts, tk_fin, mdl_fin, batch_size=8)
                finbert_means[d] = float(np.nanmean(scores)) if len(scores) > 0 else 0.0
            except Exception as e:
                if verbose: print("FinBERT scoring error:", e)
                finbert_means[d] = 0.0
    except Exception as e:
        if verbose: print("FinBERT load/score failed:", e)
        finbert_ok = False
    finally:
        if "mdl_fin" in locals():
            del mdl_fin
        if "tk_fin" in locals():
            del tk_fin
        free_torch()

    # RoBERTa scoring (FIXED: use AutoModelForSequenceClassification)
    roberta_ok = False
    roberta_means = {}
    try:
        if verbose: print("Loading RoBERTa:", roberta_model_name, "on", DEVICE)
        tk_ro, mdl_ro = load_text_classifier(roberta_model_name)
        roberta_ok = True
        for d, texts in per_day_texts.items():
            if not texts:
                roberta_means[d] = 0.0
                continue
            try:
                scores = sentiment_score_from_classifier(texts, tk_ro, mdl_ro, batch_size=8)
                roberta_means[d] = float(np.nanmean(scores)) if len(scores) > 0 else 0.0
            except Exception as e:
                if verbose: print("RoBERTa scoring error:", e)
                roberta_means[d] = 0.0
    except Exception as e:
        if verbose: print("RoBERTa load/score failed:", e)
        roberta_ok = False
    finally:
        if "mdl_ro" in locals():
            del mdl_ro
        if "tk_ro" in locals():
            del tk_ro
        free_torch()

    # Embeddings (sentence-transformers)
    embed_ok = False
    embed_norms = {}
    try:
        if verbose: print("Loading embedder:", embed_model_name)
        embedder = load_sentence_transformer(embed_model_name)
        embed_ok = True
        for d, texts in per_day_texts.items():
            if not texts:
                embed_norms[d] = 0.0
                continue
            try:
                embs = embedder.encode(texts, output_value="numpy", show_progress_bar=False)
                embed_norms[d] = float(np.linalg.norm(np.mean(embs, axis=0)))
            except Exception as e:
                if verbose: print("Embedding error:", e)
                embed_norms[d] = 0.0
    except Exception as e:
        if verbose: print("Embedder load/encode failed:", e)
        embed_ok = False
    finally:
        try:
            del embedder
        except Exception:
            pass
        free_torch()

    # aggregate days -> df_daily_sentagg
    days = []
    for d in sorted(per_day_texts.keys()):
        days.append({
            "date": pd.Timestamp(d),
            "article_count": len(per_day_texts[d]),
            "finbert_mean": finbert_means.get(d, 0.0),
            "roberta_mean": roberta_means.get(d, 0.0),
            "embed_mean_norm": embed_norms.get(d, 0.0)
        })
    df_daily_sentagg = pd.DataFrame(days).sort_values("date").reset_index(drop=True)

    # Merge with price df on trading dates
    df_price_local = df_price.copy()
    df_price_local["date_only"] = pd.to_datetime(df_price_local["date"]).dt.date
    df_daily_sentagg["date_only"] = pd.to_datetime(df_daily_sentagg["date"]).dt.date
    df_price_local["return_1"] = df_price_local["adjusted_close"].pct_change()
    df_price_local["vol_5"] = df_price_local["return_1"].rolling(5).std()
    merged = pd.merge(df_price_local, df_daily_sentagg,  how="inner")
    if merged.empty:
        return None, {"finbert": finbert_ok, "roberta": roberta_ok, "embedder": embed_ok}, {"message": "No overlap between price dates and sentiment dates."}

    merged = merged.dropna().reset_index(drop=True)

    features = ["finbert_mean", "roberta_mean", "embed_mean_norm", "article_count", "return_1", "vol_5"]

    # Ensure 'target_next' exists
    if "target_next" not in merged.columns:
        merged = merged.sort_values("date_only")
        merged["target_next"] = merged["adjusted_close"].shift(-1)
    
    merged.rename(columns={"date_x": "date"}, inplace=True)

    df_sent_features = merged[["date", "date_only"] + features + ["adjusted_close", "target_next"]].copy()
    df_sent_features.rename(columns={"date": "date_dt"}, inplace=True)

    availability = {"finbert": finbert_ok, "roberta": roberta_ok, "embedder": embed_ok}
    return df_sent_features, availability, None

# -------------------------- Ensembling helpers -----------------------------

def inverse_rmse_weights(metrics: dict, models: list):
    """Compute weights proportional to 1/rmse for given model names. Returns dict of weights."""
    invs = []
    for m in models:
        rmse = metrics.get(m, {}).get("rmse", None)
        if rmse is None:
            invs.append(None)
            continue
        try:
            rmse = float(rmse)
            if rmse == 0 or np.isnan(rmse):
                invs.append(None)
            else:
                invs.append(1.0 / rmse)
        except:
            invs.append(None)

    # replace None with equal small weight (will be normalized)
    valid = [v for v in invs if v is not None]
    if not valid:
        # all invalid -> equal weights
        n = len(models)
        return {m: 1.0 / n for m in models}
    
    # set missing to mean of valid small value
    mean_inv = float(np.mean(valid))
    invs = [v if v is not None else mean_inv for v in invs]
    arr = np.array(invs, dtype=float)
    weights = arr / arr.sum()
    return {models[i]: float(weights[i]) for i in range(len(models))}

def price_ensemble_predict(price_models: dict, last_rows_df: pd.DataFrame, features: list, metrics: dict, seq_window: int = 30) -> Tuple[float, dict]:
    """Make per-model price predictions and compute weighted ensemble using inverse-RMSE weights."""
    tab_input = np.nan_to_num(tab_input, nan=0.0, posinf=0.0, neginf=0.0)
    tab_input = last_rows_df[features].values[-1].reshape(1, -1)
    seq_input = last_rows_df[features].values[-seq_window:].reshape(1, seq_window, len(features), 1)

    preds = {}
    # LSTM
    try:
        preds["lstm"] = float(price_models["lstm"].predict(seq_input)[0, 0])
    except Exception:
        preds["lstm"] = None
    # CNN-LSTM
    try:
        preds["cnn_lstm"] = float(price_models["cnn_lstm"].predict(seq_input)[0, 0])
    except Exception:
        preds["cnn_lstm"] = None
    # LightGBM
    try:
        preds["lightgbm"] = float(price_models["lightgbm"].predict(tab_input)[0])
    except Exception:
        preds["lightgbm"] = None
    # RandomForest
    try:
        preds["random_forest"] = float(price_models["random_forest"].predict(tab_input)[0])
    except Exception:
        preds["random_forest"] = None

    # Compute weights from metrics
    model_list = ["lstm", "cnn_lstm", "lightgbm", "random_forest"]
    weights = inverse_rmse_weights(metrics, model_list)

    # Build ensemble from available preds
    avail_preds = []
    avail_weights = []
    for m in model_list:
        p = preds.get(m, None)
        w = weights.get(m, 0.0)
        if p is not None:
            avail_preds.append(p)
            avail_weights.append(w)
    if not avail_preds:
        raise RuntimeError("No price model predictions available.")
    # normalize weights among available models
    w_arr = np.array(avail_weights, dtype=float)
    w_arr = w_arr / w_arr.sum()
    ensemble = float(np.sum(np.array(avail_preds) * w_arr))
    # return ensemble and raw preds
    return ensemble, preds

def sentiment_ensemble_predict(df_sent_features: pd.DataFrame, embed_xgb_model=None) -> Tuple[float, dict]:
    """
    Build sentiment ensemble effect for latest day.
    - FinBERT and RoBERTa produce mean scores ~[-1,1] and are mapped to dollar residuals here.
    - Embedding xgb model predicts residual directly if available.
    Returns (sentiment_effect_in_dollars, per_component_effects)
    """
    if df_sent_features is None or df_sent_features.empty:
        return 0.0, {"message": "no_sent_features"}

    latest = df_sent_features.iloc[-1]
    adj_close = float(latest["adjusted_close"])
    fin = float(latest.get("finbert_mean", 0.0))
    ro = float(latest.get("roberta_mean", 0.0))
    emb_norm = float(latest.get("embed_mean_norm", 0.0))
    count = float(latest.get("article_count", 0.0))
    ret1 = float(latest.get("return_1", 0.0))
    vol5 = float(latest.get("vol_5", 0.0))

    # Map classifier means to dollar residuals: scaling heuristic relative to price
    scale = 0.001 * np.log1p(count)
    fin_effect = fin * (adj_close * scale)
    ro_effect = ro * (adj_close * scale)

    if embed_xgb_model is not None:
        X_emb = np.array([[emb_norm/50, count/20, ret1, vol5]])
        try:
            emb_effect = float(embed_xgb_model.predict(X_emb)[0])
        except Exception:
            emb_effect = 0.0
    else:
        emb_effect = 0.0

    # simple average of components (you could weight by validation RMSEs if available)
    comps = {"finbert_effect": fin_effect, "roberta_effect": ro_effect, "embed_effect": emb_effect}
    effects = [v for v in comps.values()]
    ensemble = np.average(effects, weights=[1,1,(embed_xgb_model is not None)])
    return ensemble, comps

# -------------------------- Full pipeline ---------------------------------

def advance_analysis(ticker: str,
                     seq_window: int = 30,
                     train_lstm_flag: bool = True,
                     train_cnn_flag: bool = True,
                     train_lgb_flag: bool = True,
                     train_rf_flag: bool = True,
                     train_sent_embed_flag: bool = True,
                     finbert_model_name: str = "yiyanghkust/finbert-tone",
                     roberta_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
                     embed_model_name: str = "all-MiniLM-L6-v2",
                     w_price: float = 0.7,
                     w_sent: float = 0.3,
                     verbose: bool = True) -> Dict:
    """
    Orchestrator: fetch price, train price models, build sentiment features, train embed->xgb,
    compute ensembles, return metrics and next-day prediction.
    NOTE: train_lgb_flag / train_rf_flag currently do not skip training inside train_price_models;
    they only control whether those models are *used* for downstream embed baseline / outputs.
    To skip training them entirely (faster), modify train_price_models accordingly.
    """
    start_time = time.time()
    df_price_local = fetch_intraday_or_daily(ticker, interval="60min", use_intraday=False)

    if df_price_local is None or df_price_local.empty:
        raise RuntimeError("No price data fetched")

    # Ensure date column is datetime
    if "date" in df_price_local.columns:
        df_price_local["date"] = pd.to_datetime(df_price_local["date"])
    else:
        # try to fall back to index or raise
        try:
            df_price_local["date"] = pd.to_datetime(df_price_local.index)
        except Exception:
            raise RuntimeError("Price data has no 'date' column and index is not datetime")

    # Train price models (LSTM/CNN epochs controlled by flags)
    price_models, holdout, price_metrics, df_prepared, price_features = train_price_models(
        df_price_local,
        seq_window=seq_window,
        epochs_lstm=(10 if train_lstm_flag else 0),
        epochs_cnnlstm=(8 if train_cnn_flag else 0),
        verbose=(1 if verbose else 0)
    )
    if verbose: print("Price metrics:", price_metrics)

    # Fetch sentiment articles
    df_sent_articles = fetch_sentiment_feed(ticker)

    # Build sentiment aggregated features (no global variable usage)
    sent_features_df, sent_availability_info, _ = build_sentiment_ensemble(
        df_price=df_price_local,
        df_sent_articles=df_sent_articles,
        finbert_model_name=finbert_model_name,
        roberta_model_name=roberta_model_name,
        embed_model_name=embed_model_name,
        verbose=(1 if verbose else 0)
    )
    if verbose: print("Sentiment availability:", sent_availability_info)

    # Train embedding -> residual XGBoost if enough data and sentiment features exist
    embed_xgb = None
    sentiment_metrics = {"embed_xgb": None}
    if sent_features_df is not None and len(sent_features_df) >= 30 and train_sent_embed_flag:
        # compute engineered price columns on full price DataFrame
        df_price_map = ensure_price_engineered(df_price_local.copy())
        # Ensure date_only exists in both sides as date objects
        sent_features_df["date_only"] = pd.to_datetime(sent_features_df["date_dt"]).dt.date
        df_price_map["date_only"] = pd.to_datetime(df_price_map["date"]).dt.date

        # merge on date_only preserving price-aligned features
        merged_for_embed = pd.merge(sent_features_df, df_price_map, on="date_only", how="inner", suffixes=("_sent", "_price"))
        merged_for_embed = merged_for_embed.sort_values("date_only").reset_index(drop=True)

        if len(merged_for_embed) >= 30:
            # Build X for embed model (normalize emb_norm a bit to avoid huge scale)
            X_tab_for_embed = merged_for_embed[["embed_mean_norm", "article_count", "return_1", "vol_5"]].fillna(0).values.astype(np.float32)
            # Simple normalization for embed_norm and article_count to keep scales comparable
            # (these factors are arbitrary but stabilize training)
            if X_tab_for_embed.shape[0] > 0:
                X_tab_for_embed[:, 0] = X_tab_for_embed[:, 0] / max(1.0, np.percentile(X_tab_for_embed[:, 0], 80))  # embed_norm
                X_tab_for_embed[:, 1] = X_tab_for_embed[:, 1] / np.maximum(1.0, np.percentile(X_tab_for_embed[:, 1], 80))  # article_count

            # Build baseline predictions (average of LGB & RF) while respecting train_lgb_flag / train_rf_flag
            baseline_tab = []
            for _, row in merged_for_embed.iterrows():
                # extract tab features in the same order as price_features
                try:
                    tab_row = row[price_features].values.reshape(1, -1).astype(np.float32)
                except Exception:
                    # fallback: try selecting from price side (in case of suffixes)
                    tab_row = row[[c for c in price_features if c in row.index]].values.reshape(1, -1).astype(np.float32)
                # guard against NaNs
                tab_row = np.nan_to_num(tab_row, nan=0.0, posinf=0.0, neginf=0.0)

                preds = []
                if train_lgb_flag and ("lightgbm" in price_models and price_models["lightgbm"] is not None):
                    try:
                        preds.append(float(price_models["lightgbm"].predict(tab_row)[0]))
                    except Exception:
                        preds.append(np.nan)
                if train_rf_flag and ("random_forest" in price_models and price_models["random_forest"] is not None):
                    try:
                        preds.append(float(price_models["random_forest"].predict(tab_row)[0]))
                    except Exception:
                        preds.append(np.nan)
                # if both disabled, fall back to using average of sequential models where available (not ideal)
                if not preds:
                    # try LSTM/CNN preds using last seq_window rows around this date if available
                    preds = [np.nan]

                baseline_tab.append(np.nanmean(preds))
            baseline_tab = np.array(baseline_tab, dtype=np.float32)

            # Align target_next: expect price-based target (next day's adjusted_close)
            # If target_next missing, attempt to create it from price side
            if "target_next" not in merged_for_embed.columns:
                # try to create from adjusted_close_price column
                if "adjusted_close" in merged_for_embed.columns:
                    merged_for_embed = merged_for_embed.sort_values("date_only")
                    merged_for_embed["target_next"] = merged_for_embed["adjusted_close"].shift(-1).values
                else:
                    raise RuntimeError("Cannot find 'target_next' or 'adjusted_close' for residual computation.")

            # Remove rows where target_next is NaN (last day)
            valid_idx = ~np.isnan(baseline_tab) & ~np.isnan(merged_for_embed["target_next"].values.astype(np.float32))
            if valid_idx.sum() >= 20:
                X_tab_train = X_tab_for_embed[valid_idx]
                baseline_valid = baseline_tab[valid_idx]
                residuals = merged_for_embed.loc[valid_idx, "target_next"].values.astype(np.float32) - baseline_valid

                # Train XGBoost on residuals
                embed_xgb = train_xgboost_regressor(X_tab_train, residuals)
                preds_embed = embed_xgb.predict(X_tab_train)
                sentiment_metrics["embed_xgb"] = {
                    "rmse": float(np.sqrt(mean_squared_error(residuals, preds_embed))),
                    "r2": float(r2_score(residuals, preds_embed))
                }
            else:
                if verbose: print("Not enough valid rows for embed->xgb training after alignment:", int(valid_idx.sum()))

    # Price ensemble holdout evaluation (simple average baseline of lgb & rf)
    try:
        y_tab_test = holdout["tab_y_test"]
        p_lgb = holdout.get("lgb_pred", np.full_like(y_tab_test, np.nan))
        p_rf = holdout.get("rf_pred", np.full_like(y_tab_test, np.nan))
        combined_price_holdout_preds = np.nanmean(np.vstack([p_lgb, p_rf]), axis=0)
        combined_price_rmse = float(np.sqrt(mean_squared_error(y_tab_test, combined_price_holdout_preds)))
        combined_price_r2 = float(r2_score(y_tab_test, combined_price_holdout_preds))
    except Exception:
        combined_price_rmse = None
        combined_price_r2 = None

    # Prepare last_rows for sequence/tab predictions
    if df_prepared is None or df_prepared.shape[0] < seq_window:
        raise RuntimeError("Not enough prepared price rows to create sequence for prediction.")
    last_rows_df = df_prepared.tail(seq_window + 1).reset_index(drop=True)

    # Price ensemble prediction
    price_ens, price_model_preds = price_ensemble_predict(price_models, last_rows_df, price_features, price_metrics, seq_window=seq_window)

    # Sentiment ensemble prediction
    sent_ens = 0.0
    sent_model_preds = {}
    if sent_features_df is not None:
        sent_ens, sent_model_preds = sentiment_ensemble_predict(sent_features_df, embed_xgb_model=embed_xgb)

    final_pred = float(w_price * price_ens + w_sent * sent_ens)

    elapsed = time.time() - start_time
    result = {
        "ticker": ticker,
        "elapsed_seconds": elapsed,
        "price_ensemble": price_ens,
        "price_model_preds": price_model_preds,
        "price_models_metrics": price_metrics,
        "combined_price_holdout_rmse": combined_price_rmse,
        "combined_price_holdout_r2": combined_price_r2,
        "sentiment_ensemble": sent_ens,
        "sentiment_model_preds": sent_model_preds,
        "sentiment_metrics": sentiment_metrics,
        "final_pred": final_pred,
    }
    return result

