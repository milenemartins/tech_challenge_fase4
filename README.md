# Tech Challenge Fase 4 — LSTM Stock Price Prediction

Projeto desenvolvido para o Tech Challenge da Fase 4 da Pós-Tech em Machine Learning Engineering (FIAP/Alura).

## Objetivo

Criar um modelo preditivo de redes neurais **LSTM (Long Short-Term Memory)** para prever o valor de fechamento das ações da **Disney (DIS)** na bolsa de valores, com deploy em uma API RESTful.

---

## Arquitetura do Projeto

```
tech_challenge_fase4/
├── data/                  # Dados históricos coletados
├── model/                 # Modelo LSTM treinado (.keras)
├── notebooks/             # Notebooks de exploração e treinamento
├── src/                   # Scripts de coleta, pré-processamento e treino
│   ├── collect_data.py    # Coleta de dados via yfinance
│   ├── preprocess.py      # Normalização e criação de sequências
│   └── train.py           # Treinamento e avaliação do modelo LSTM
├── api/                   # API RESTful com FastAPI
│   ├── main.py            # Endpoints da API
│   └── schemas.py         # Schemas de entrada e saída (Pydantic)
├── monitoring/            # Logs e métricas de monitoramento
├── tests/                 # Testes unitários
├── Dockerfile             # Container da API
├── docker-compose.yml     # Orquestração dos serviços
├── requirements.txt       # Dependências do projeto
└── .gitignore
```

---

## Stack

| Componente      | Tecnologia              |
|-----------------|-------------------------|
| Dados           | yfinance                |
| Modelo          | Keras / TensorFlow      |
| API             | FastAPI                 |
| Containerização | Docker / Docker Compose |
| Deploy          | Railway / Render        |
| Monitoramento   | Logs via middleware FastAPI |

---

## Como Rodar Localmente

### Pré-requisitos

- Python 3.10+
- Docker e Docker Compose instalados

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd tech_challenge_fase4
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Coletar dados e treinar o modelo

```bash
python src/collect_data.py
python src/train.py
```

### 4. Rodar a API localmente

```bash
uvicorn api.main:app --reload
```

Acesse a documentação em: `http://localhost:8000/docs`

### 5. Rodar com Docker

```bash
docker-compose up --build
```

---

## Endpoints da API

| Método | Endpoint     | Descrição                                  |
|--------|--------------|--------------------------------------------|
| GET    | `/`          | Health check da API                        |
| POST   | `/predict`   | Recebe dados históricos e retorna previsão |
| GET    | `/metrics`   | Retorna métricas do modelo (MAE, RMSE, MAPE) |

---

## Métricas de Avaliação

O modelo é avaliado com as seguintes métricas:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)

---

## Dados

- **Empresa:** Disney (DIS)
- **Fonte:** Yahoo Finance via `yfinance`
- **Período:** 2018-01-01 até 2024-07-20
- **Feature alvo:** `Close` (preço de fechamento)

---

## Entregáveis

- [x] Código-fonte do modelo LSTM
- [x] Documentação do projeto (este README)
- [ ] Scripts Docker para deploy
- [ ] Link da API em produção
- [ ] Vídeo demonstrando o funcionamento

---

## Autora

Milene Martins — Pós-Tech Machine Learning Engineering (FIAP/Alura)
