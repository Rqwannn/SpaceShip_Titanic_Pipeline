import os
import time
import json
import httpx
import joblib
import psutil
import pandas as pd
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Gauge, Histogram, Summary

artifacts = {}

def load_artifacts():
    global artifacts
    artifact_files = [f for f in os.listdir(ARTIFACTS_PATH) if f.endswith('.pkl')]
    if not artifact_files:
        raise RuntimeError(f"Tidak ada file artefak (.pkl) yang ditemukan di {ARTIFACTS_PATH}")
    
    for file in artifact_files:
        key = file.replace('.pkl', '')
        with open(os.path.join(ARTIFACTS_PATH, file), 'rb') as f:
            artifacts[key] = joblib.load(f)
    print(f"Berhasil memuat {len(artifacts)} artefak.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(
    title="ML Model Monitoring",
    description="Proxy service untuk memonitor ML model dengan FastAPI & Prometheus.",
    version="1.0.0",
    lifespan=lifespan
)

MODEL_URI = os.getenv('MODEL_URI', 'http://localhost:8080/invocations')
ARTIFACTS_PATH = os.getenv('ARTIFACTS_PATH', './artifacts')

REQUESTS_TOTAL = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
RESPONSES_TOTAL = Counter('http_responses_total', 'Total HTTP responses', ['method', 'endpoint', 'status_code'])
REQUEST_LATENCY = Histogram('http_request_latency_seconds', 'Distribusi latensi request (detik)', ['method', 'endpoint'])
REQUEST_PAYLOAD_SIZE = Histogram('request_payload_size_bytes', 'Distribusi ukuran payload request (bytes)')
IN_PROGRESS_REQUESTS = Gauge('in_progress_requests', 'Jumlah request yang sedang diproses')
PREDICTIONS_TOTAL = Counter('predictions_total', 'Total prediksi berdasarkan kelas hasil', ['prediction_class'])
PREDICTION_CONFIDENCE = Summary('prediction_confidence_score', 'Distribusi skor kepercayaan model')
INTERNAL_ERRORS_TOTAL = Counter('internal_server_errors_total', 'Total error internal server')
HEALTH_CHECKS_TOTAL = Counter('health_checks_total', 'Total health checks yang dilakukan')
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'Penggunaan CPU sistem saat ini (%)')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'Penggunaan memori sistem saat ini (%)')

class PassengerInput(BaseModel):
    home_planet: str
    deck: str
    side: str
    age: float
    cabin_num: float
    room_service: float
    food_court: float
    shopping_mall: float
    spa: float
    vr_deck: float

def preprocess_data(payload: PassengerInput) -> pd.DataFrame:
    df = pd.DataFrame([payload.model_dump()])
    
    NAME_MAP = {
        'home_planet': 'HomePlanet', 'deck': 'Deck', 'side': 'Side', 'age': 'Age',
        'cabin_num': 'CabinNum', 'room_service': 'RoomService', 'food_court': 'FoodCourt',
        'shopping_mall': 'ShoppingMall', 'spa': 'Spa', 'vr_deck': 'VRDeck', 'total_spend': 'TotalSpend'
    }

    df['total_spend'] = df[['room_service', 'food_court', 'shopping_mall', 'spa', 'vr_deck']].sum(axis=1)

    for cat_col_lower in ['home_planet', 'deck', 'side']:
        col_pascal = NAME_MAP[cat_col_lower]
        encoder = artifacts[f"{col_pascal}_encoder"]
        df[col_pascal] = encoder.transform(df[[cat_col_lower]])

    cols_to_scale_lower = ['age', 'room_service', 'food_court', 'shopping_mall', 'spa', 'vr_deck', 'cabin_num', 'total_spend']
    for col_lower in cols_to_scale_lower:
        col_pascal = NAME_MAP[col_lower]
        scaler = artifacts[f"{col_pascal}_scaler"]
        
        temp_df_for_scaling = df[[col_lower]].rename(columns={col_lower: col_pascal})
        df[col_pascal] = scaler.transform(temp_df_for_scaling)
    
    final_cols_order = [
        'HomePlanet', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall',
        'Spa', 'VRDeck', 'Deck', 'CabinNum', 'Side', 'TotalSpend'
    ]
    
    return df[final_cols_order]

app.mount("/metrics", make_asgi_app())

@app.get("/health")
async def health_check():
    HEALTH_CHECKS_TOTAL.inc()
    return {"status": "ok"}

@app.post("/invocations")
async def invocations(payload: PassengerInput, request: Request, response: Response):
    start_time = time.time()
    IN_PROGRESS_REQUESTS.inc()
    REQUESTS_TOTAL.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_PAYLOAD_SIZE.observe(len(await request.body()))
    
    status_code = 500
    response_data = {"error": "Internal server error"}

    try:
        processed_df = preprocess_data(payload)
        model_payload = {"dataframe_split": processed_df.to_dict(orient='split', index=False)}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            model_response = await client.post(MODEL_URI, json=model_payload)
        
        model_response.raise_for_status()
        prediction_result = model_response.json()
        
        predictions = prediction_result.get('predictions', [])
        for p in predictions:
            PREDICTIONS_TOTAL.labels(prediction_class=str(p)).inc()

        status_code = model_response.status_code
        response_data = prediction_result

    except Exception as e:
        INTERNAL_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(latency)
        RESPONSES_TOTAL.labels(method=request.method, endpoint=request.url.path, status_code=status_code).inc()
        IN_PROGRESS_REQUESTS.dec()
        
        SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
        SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().percent)
        
        response.status_code = status_code

    return response_data