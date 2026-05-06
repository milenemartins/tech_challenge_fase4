import os
import json
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import joblib
import psutil
from keras.models import load_model
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.schemas import PredictRequest, PredictResponse, MetricsResponse, HealthResponse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, 'model', 'lstm_dis_best.keras')
SCALER_PATH  = os.path.join(BASE_DIR, 'data',  'scaler.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'model', 'metrics.json')
META_PATH    = os.path.join(BASE_DIR, 'data',  'preprocessing_metadata.json')
LOG_PATH     = os.path.join(BASE_DIR, 'monitoring', 'requests.log')

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Estado global da aplicação
# ---------------------------------------------------------------------------
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Carregando modelo e artefatos...")
    app_state['model']  = load_model(MODEL_PATH)
    app_state['scaler'] = joblib.load(SCALER_PATH)

    with open(METRICS_PATH) as f:
        app_state['metrics'] = json.load(f)

    with open(META_PATH) as f:
        app_state['meta'] = json.load(f)

    logger.info("Modelo carregado com sucesso.")
    yield
    app_state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LSTM Stock Price Prediction API",
    description="API para previsão do preço de fechamento da ação da Disney (DIS) usando modelo LSTM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware de monitoramento
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    start      = time.perf_counter()
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().percent

    response = await call_next(request)

    elapsed_ms  = round((time.perf_counter() - start) * 1000, 2)
    cpu_after   = psutil.cpu_percent(interval=None)
    mem_after   = psutil.virtual_memory().percent
    process_mem = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)

    log_entry = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "method":           request.method,
        "endpoint":         str(request.url.path),
        "status_code":      response.status_code,
        "duration_ms":      elapsed_ms,
        "cpu_before_%":     cpu_before,
        "cpu_after_%":      cpu_after,
        "ram_system_%":     mem_after,
        "ram_process_mb":   process_mem,
    }

    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

    response.headers["X-Response-Time-ms"] = str(elapsed_ms)
    response.headers["X-RAM-Process-MB"]   = str(process_mem)
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Verifica se a API está online e o modelo carregado."""
    return HealthResponse(
        status="ok",
        model_loaded="model" in app_state,
        symbol=app_state.get('meta', {}).get('symbol', 'DIS'),
        window_size=app_state.get('meta', {}).get('window_size', 60),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Predição"])
def predict(payload: PredictRequest):
    """
    Recebe uma lista de preços históricos de fechamento (mínimo 60 valores)
    e retorna o preço de fechamento previsto para o próximo dia.
    """
    model   = app_state.get('model')
    scaler  = app_state.get('scaler')
    meta    = app_state.get('meta', {})

    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")

    window = meta.get('window_size', 60)

    # Usa apenas os últimos `window` valores
    prices = np.array(payload.prices[-window:], dtype=np.float32).reshape(-1, 1)

    prices_scaled = scaler.transform(prices)
    X = prices_scaled.reshape(1, window, 1)

    pred_scaled   = model.predict(X, verbose=0)
    predicted_price = float(scaler.inverse_transform(pred_scaled)[0][0])

    return PredictResponse(
        symbol=meta.get('symbol', 'DIS'),
        predicted_price=round(predicted_price, 4),
        window_size=window,
        input_length=len(payload.prices),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoramento"])
def get_metrics():
    """Retorna as métricas de avaliação do modelo (MAE, RMSE, MAPE) no conjunto de teste."""
    metrics = app_state.get('metrics')
    if metrics is None:
        raise HTTPException(status_code=503, detail="Métricas não disponíveis.")
    return MetricsResponse(**metrics)


@app.get("/monitoring", tags=["Monitoramento"])
def get_monitoring():
    """Retorna o uso atual de CPU, memória do sistema e memória do processo."""
    process = psutil.Process(os.getpid())
    mem     = psutil.virtual_memory()
    disk    = psutil.disk_usage('/')
    return {
        "cpu_percent":        psutil.cpu_percent(interval=0.1),
        "ram_system_percent": mem.percent,
        "ram_system_used_mb": round(mem.used / 1024 / 1024, 2),
        "ram_system_total_mb": round(mem.total / 1024 / 1024, 2),
        "ram_process_mb":     round(process.memory_info().rss / 1024 / 1024, 2),
        "disk_used_gb":       round(disk.used / 1024 / 1024 / 1024, 2),
        "disk_total_gb":      round(disk.total / 1024 / 1024 / 1024, 2),
        "disk_percent":       disk.percent,
    }
