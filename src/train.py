"""Treina o modelo LSTM e salva os artefatos em model/."""

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def build_model(window_size: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mean_squared_error",
    )
    return model


def main():
    for path in [os.path.join(DATA_DIR, f) for f in ("X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}. Execute preprocess.py primeiro.")

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))

    with open(os.path.join(DATA_DIR, "preprocessing_metadata.json")) as f:
        meta = json.load(f)

    window_size = meta["window_size"]
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"TensorFlow: {tf.__version__}")
    print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    model = build_model(window_size)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "lstm_dis_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )

    print(f"\nMelhor val_loss : {min(history.history['val_loss']):.6f}")
    print(f"Épocas treinadas: {len(history.history['loss'])}")

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mape = float(np.mean(np.abs((y_real - y_pred) / y_real)) * 100)

    metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}
    print("\n=== Métricas no Conjunto de Teste ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    model.save(os.path.join(MODEL_DIR, "lstm_dis_final.keras"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(os.path.join(MODEL_DIR, "y_pred.npy"), y_pred)
    np.save(os.path.join(MODEL_DIR, "y_real.npy"), y_real)

    # Lê datas do CSV para o gráfico de previsões
    df_raw = pd.read_csv(os.path.join(DATA_DIR, "dis_raw.csv"), index_col=0, parse_dates=True).dropna()
    test_index = df_raw.index[-len(y_real):]

    try:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-darkgrid")

        plt.figure(figsize=(12, 4))
        plt.plot(history.history["loss"], label="Treino", color="steelblue")
        plt.plot(history.history["val_loss"], label="Validação", color="coral")
        plt.title("Curvas de Aprendizado — Loss (MSE)")
        plt.xlabel("Época")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "learning_curves.png"), dpi=120)
        plt.close()

        plt.figure(figsize=(14, 5))
        plt.plot(test_index, y_real, label="Preço Real", color="steelblue", linewidth=1.2)
        plt.plot(test_index, y_pred, label="Preço Previsto", color="coral", linewidth=1.2, linestyle="--")
        plt.title(f"Previsão vs Realidade — Disney (DIS)\nMAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
        plt.ylabel("Preço (USD)")
        plt.xlabel("Data")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "predictions_vs_real.png"), dpi=120)
        plt.close()

        print("Gráficos salvos em model/")
    except ImportError:
        pass

    print("Artefatos salvos em model/")


if __name__ == "__main__":
    main()
