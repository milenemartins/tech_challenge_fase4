"""Normaliza os dados brutos e cria sequências de entrada para o LSTM."""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 60
TRAIN_RATIO = 0.80
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def create_sequences(data: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def main():
    raw_path = os.path.join(DATA_DIR, "dis_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {raw_path}. Execute collect_data.py primeiro.")

    df_raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    df_raw = df_raw.dropna()

    df = df_raw[["Close"]].copy().sort_index()
    df = df[~df.index.duplicated(keep="first")].dropna()

    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"Treino : {len(train_df)} registros  ({train_df.index.min().date()} → {train_df.index.max().date()})")
    print(f"Teste  : {len(test_df)}  registros  ({test_df.index.min().date()} → {test_df.index.max().date()})")

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df["Close"].values.reshape(-1, 1))
    test_scaled = scaler.transform(test_df["Close"].values.reshape(-1, 1))

    X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE)

    # Inclui os últimos WINDOW_SIZE dias do treino no teste para não perder contexto
    test_full = np.concatenate([train_scaled[-WINDOW_SIZE:], test_scaled], axis=0)
    X_test, y_test = create_sequences(test_full, WINDOW_SIZE)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(f"X_train: {X_train.shape}  |  y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}   |  y_test : {y_test.shape}")

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
    joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.pkl"))

    metadata = {
        "symbol": "DIS",
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "window_size": WINDOW_SIZE,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "train_ratio": TRAIN_RATIO,
        "scaler": "MinMaxScaler(0,1)",
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
    }
    with open(os.path.join(DATA_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Artefatos salvos em data/")


if __name__ == "__main__":
    main()
