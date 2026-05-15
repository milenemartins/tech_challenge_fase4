FROM python:3.9-slim

WORKDIR /app

# Instala dependências do sistema necessárias para o TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/     ./api/
COPY src/     ./src/
COPY model/   ./model/
COPY data/    ./data/

ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
