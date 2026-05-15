"""Coleta dados históricos da Disney (DIS) via Yahoo Finance API e salva em data/dis_raw.csv."""

import json
import os
import subprocess
from datetime import datetime

import pandas as pd

SYMBOL = "DIS"
START_DATE = "2018-01-01"
END_DATE = "2024-07-20"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    period1 = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    period2 = int(datetime.strptime(end, "%Y-%m-%d").timestamp())

    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&period1={period1}&period2={period2}"
    )

    result = subprocess.run(
        [
            "curl", "-s", "-L", url,
            "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "-H", "Accept: application/json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    data = json.loads(result.stdout)
    chart = data["chart"]["result"][0]
    quotes = chart["indicators"]["quote"][0]

    df = pd.DataFrame(
        {
            "Open": quotes["open"],
            "High": quotes["high"],
            "Low": quotes["low"],
            "Close": quotes["close"],
            "Volume": quotes["volume"],
        },
        index=pd.to_datetime(chart["timestamp"], unit="s").normalize(),
    )
    df.index.name = "Date"
    return df.dropna()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Coletando dados de {SYMBOL} ({START_DATE} → {END_DATE})...")
    df = fetch_ohlcv(SYMBOL, START_DATE, END_DATE)

    out_path = os.path.join(DATA_DIR, "dis_raw.csv")
    df.to_csv(out_path)

    print(f"Shape: {df.shape}")
    print(f"Período: {df.index.min().date()} → {df.index.max().date()}")
    print(f"Salvo em: {out_path}")


if __name__ == "__main__":
    main()
