# -*- coding: utf-8 -*-
"""
Analiza 10 compañías con yfinance, entrena un MLP simple por cada ticker,
estima probabilidad de subir mañana, calcula riesgo (volatilidad diaria)
y genera un ranking descendente. Exporta resultados a ranking.json incluyendo OHLC.
"""

import json
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ============ CONFIG ============
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
           "META", "NVDA", "NFLX", "ADBE", "INTC"]
PERIOD = "1mo"     # último mes
INTERVAL = "1d"    # diario
EPOCHS = 100
BATCH_SIZE = 1

def recommendation_from_prob(p_up: float) -> str:
    if p_up > 0.60:
        return "Comprar"
    elif p_up >= 0.50:
        return "Mantener"
    else:
        return "Evitar"

def risk_from_vol(vol: float) -> str:
    if vol < 0.010:
        return "Bajo"
    elif vol < 0.020:
        return "Medio"
    else:
        return "Alto"

def analyze_stock(symbol: str):
    # 1) Descargar datos
    df = yf.download(symbol, period=PERIOD, interval=INTERVAL, progress=False)
    if df.empty or "Close" not in df.columns:
        return None

    df = df.dropna()
    df["ret"] = df["Close"].pct_change()
    if df["Close"].shape[0] < 5:
        return None
    vol = float(df["ret"].std(skipna=True))  # volatilidad diaria

    # Preparar datos para MLP
    prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(len(prices_scaled) - 1):
        X.append(prices_scaled[i])
        y.append(1 if prices_scaled[i + 1] > prices_scaled[i] else 0)
    X = np.array(X)
    y = np.array(y)

    # Modelo MLP básico
    model = Sequential()
    model.add(Dense(10, input_dim=1, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    # Predicción para el último dato
    last_price = prices_scaled[-1].reshape(1, -1)
    prob_up = float(model.predict(last_price, verbose=0)[0][0])

    # Extraer OHLC para gráficas de velas
    ohlc_list = []
    for date, row in df.iterrows():
        ohlc_list.append({
            "date": str(date.date()),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        })

    return {
        "symbol": symbol,
        "last_close": float(df["Close"].iloc[-1]),
        "probability_up": prob_up,
        "risk_daily_vol": vol,
        "risk_level": risk_from_vol(vol),
        "recommendation": recommendation_from_prob(prob_up),
        "ohlc": ohlc_list  # <--- aquí está el OHLC completo
    }

def main():
    results = []
    for t in TICKERS:
        try:
            r = analyze_stock(t)
            if r:
                results.append(r)
            else:
                print(f"[WARN] Sin datos suficientes para {t}")
        except Exception as e:
            print(f"[ERROR] {t}: {e}")

    results_sorted = sorted(results, key=lambda x: x["probability_up"], reverse=True)

    print("\n===== Ranking de inversión (mayor a menor) =====")
    for i, r in enumerate(results_sorted, 1):
        print(
            f"{i}. {r['symbol']} | ProbUp: {r['probability_up']:.2f} | "
            f"Riesgo: {r['risk_level']} (vol={r['risk_daily_vol']:.3%}) | "
            f"UltCierre: {r['last_close']:.2f} | Reco: {r['recommendation']}"
        )

    payload = {
        "universe": TICKERS,
        "period": PERIOD,
        "interval": INTERVAL,
        "ranking": results_sorted,
    }
    with open("ranking.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nJSON guardado en ranking.json con OHLC para gráficos de velas")

if __name__ == "__main__":
    main()
