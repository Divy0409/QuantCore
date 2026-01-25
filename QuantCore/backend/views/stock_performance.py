import time
import numpy as np
import requests
import os
from dotenv import load_dotenv
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.shortcuts import render

load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

ALPHA_BASE_URL = "https://www.alphavantage.co/query"

def stock_performance_page(request):
    return render(request, "stock_performance.html")

def fetch_latest_indicator(function_name, ticker, interval="daily", extra_params=None,return_series=False,lookback=5):
    """
    Fetch the most recent indicator value from Alpha Vantage
    """
    params = {
        "function": function_name,
        "symbol": ticker,
        "interval": interval,
        "apikey": API_KEY,
    }

    if extra_params:
        params.update(extra_params)

    response = requests.get(ALPHA_BASE_URL, params=params)
    data = response.json()

    # API limit / error handling
    if "Note" in data or "Error Message" in data:
        return {
            "available": False,
            "reason": data.get("Note") or data.get("Error Message")
        }

    indicator_key = next(
        (k for k in data.keys() if "Technical Analysis" in k),
        None
    )

    if not indicator_key or not data[indicator_key]:
        return {
            "available": False,
            "reason": "Indicator data missing"
        }

    series = data[indicator_key]

    if not series:
        return {
            "available": False,
            "reason": "Indicator data missing"
        }
    
    sorted_dates = sorted(series.keys(), reverse=True)
    if return_series:
        recent_valures = []
        for d in  sorted_dates[:lookback]:
            recent_valures.append({
                "timestamp": d,
                "values": series[d]
            })
        return {
            "available": True,
            "data": recent_valures
        }

    # Get most recent timestamp
    latest_timestamp = sorted(data[indicator_key].keys(), reverse=True)[0]
    latest_data = data[indicator_key][latest_timestamp]

    return {
        "available": True,
        "timestamp": latest_timestamp,
        "values": latest_data
    }


def stock_performance(ticker: str):
    """
    Fetch latest RSI, AROON, MOM for stock performance page
    """
    metrics = {}

    # ======================
    # RSI
    # ======================
    rsi_data = fetch_latest_indicator(
        function_name="RSI",
        ticker=ticker,
        interval="daily",
        extra_params={"time_period": 14, "series_type": "close"}
    )

    if rsi_data.get("available"):
        metrics["rsi"] = {
            "label": "Relative Strength Index (RSI)",
            "value": float(rsi_data["values"]["RSI"]),
            "format": "number",
            "precision": 2,
            "timestamp": rsi_data["timestamp"]
        }

    time.sleep(12)  # To respect API rate limits

    # ======================
    # AROON
    # ======================
    aroon_data = fetch_latest_indicator(
        function_name="AROON",
        ticker=ticker,
        interval="daily",
        extra_params={"time_period": 14, "series_type": "close"}
    )

    if aroon_data.get("available"):
        metrics["aroon_up"] = {
            "label": "Aroon Up",
            "value": float(aroon_data["values"]["Aroon Up"]),
            "format": "percent",
            "precision": 2,
            "timestamp": aroon_data["timestamp"]
        }
        metrics["aroon_down"] = {
            "label": "Aroon Down",
            "value": float(aroon_data["values"]["Aroon Down"]),
            "format": "percent",
            "precision": 2,
            "timestamp": aroon_data["timestamp"]
        }

    time.sleep(12)  # To respect API rate limits

    # ======================
    # MOM (Momentum)
    # ======================
    mom_data = fetch_latest_indicator(
        function_name="MOM",
        ticker=ticker,
        interval="daily",
        extra_params={"time_period": 10, "series_type": "close"}
    )

    if mom_data.get("available"):
        metrics["momentum"] = {
            "label": "Momentum (MOM)",
            "value": float(mom_data["values"]["MOM"]),
            "format": "number",
            "precision": 2,
            "timestamp": mom_data["timestamp"]
        }

    time.sleep(12)  # To respect API rate limits

    # ======================
    # OBV (On-Balance Volume)
    # ======================
    obv_data = fetch_latest_indicator(
        function_name="OBV",
        ticker=ticker,
        interval="daily",
        extra_params={"series_type": "close"},
        return_series=True,
        lookback=5
    )  

    def scale_number(value):
        abs_val = abs(value)
        if abs_val >= 1_000_000_000:
            return value / 1_000_000_000, "B"
        elif abs_val >= 1_000_000:
            return value / 1_000_000, "M"
        return value, ""


    if obv_data.get("available"):
        obv_values = [
            float(item["values"]["OBV"])
            for item in obv_data["data"]
            if "OBV" in item["values"]
        ]

        if obv_values:
            weekly_avg = np.mean(obv_values)
            scaled_val, unit = scale_number(weekly_avg)

            metrics["obv_weekly_avg"] = {
                "label": "OBV (5-Day Average)",
                "value": round(scaled_val, 2),
                "unit": unit,
                "format": "number",
                "precision": 2,
                "timestamp": obv_data["data"][0]["timestamp"],
                "raw_avg": weekly_avg
            }

    time.sleep(12)  # To respect API rate limits

    # ======================
    #  ADX (Average Directional Index)
    # ======================
    adx_data = fetch_latest_indicator(
        function_name="ADX",
        ticker=ticker,
        interval="daily",
        extra_params={"time_period": 14}
    )

    if adx_data.get("available"):
        metrics["adx"] = {
            "label": "Average Directional Index (ADX)",
            "value": float(adx_data["values"]["ADX"]),
            "format": "number",
            "precision": 2,
            "timestamp": adx_data["timestamp"]
        }

    time.sleep(12)  # To respect API rate limits

    # ======================
    # EMA (Exponential Moving Average)
    # ======================
    ema_data = fetch_latest_indicator(
        function_name="EMA",
        ticker=ticker,
        interval="daily",
        extra_params={"time_period": 20, "series_type": "close"}
    )

    if ema_data.get("available"):
        metrics["ema_20"] = {
            "label": "20-Day Exponential Moving Average (EMA)",
            "value": float(ema_data["values"]["EMA"]),
            "format": "number",
            "precision": 2,
            "timestamp": ema_data["timestamp"]
        }

    time.sleep(12)  # To respect API rate limits

    # ======================
    # ATR (Average True Range)
    # ======================
    atr_data = fetch_latest_indicator(
        function_name="ATR",
        ticker=ticker,
        interval="daily",
        extra_params={"time_period": 14}
    )   

    if atr_data.get("available"):
        metrics["atr"] = {
            "label": "Average True Range (ATR)",
            "value": float(atr_data["values"]["ATR"]),
            "format": "number",
            "precision": 2,
            "timestamp": atr_data["timestamp"]
        }

    return {
        "ticker": ticker.upper(),
        "metrics": metrics
    }

@require_GET
def stock_performance_view(request):
    ticker = request.GET.get("ticker")

    if not ticker:
        return JsonResponse(
            {"error": "Ticker parameter is required"},
            status=400
        )

    try:
        data = stock_performance(ticker.strip().upper())
        return JsonResponse(data, safe=False)
    except Exception as e:
        return JsonResponse(
            {"error": "Failed to fetch stock performance", "details": str(e)},
            status=500
        )