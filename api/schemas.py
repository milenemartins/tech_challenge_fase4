from pydantic import BaseModel, Field, field_validator
from typing import List


class PredictRequest(BaseModel):
    prices: List[float] = Field(
        ...,
        min_length=60,
        description="Histórico de preços de fechamento (mínimo 60 valores, em USD)"
    )

    @field_validator('prices')
    @classmethod
    def prices_must_be_positive(cls, v):
        if any(p <= 0 for p in v):
            raise ValueError('Todos os preços devem ser maiores que zero')
        return v


class PredictResponse(BaseModel):
    symbol: str
    predicted_price: float = Field(..., description="Preço de fechamento previsto (USD)")
    currency: str = "USD"
    window_size: int = Field(..., description="Janela temporal usada para a previsão")
    input_length: int = Field(..., description="Quantidade de valores recebidos")


class MetricsResponse(BaseModel):
    MAE: float = Field(..., description="Mean Absolute Error (USD)")
    RMSE: float = Field(..., description="Root Mean Square Error (USD)")
    MAPE: float = Field(..., description="Mean Absolute Percentage Error (%)")


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    symbol: str
    window_size: int
