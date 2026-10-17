"""
ColdChain IQ API — Production FastAPI backend
Supports:
- Arrhenius Kinetic Degradation Engine
- Early-Warning Anomaly Prediction (XGBoost FINAL-1.0)
- Dynamic FEFO Rankings & Audit Trails
- Sensor Telemetry Stream & Simulation
"""
import io
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- Configuration & Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "assets")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ASSETS_DIR / "coldchain_early_warning_FINAL.pkl")))

BATCHES_CSV = ASSETS_DIR / "batches.csv"
PRODUCTS_CSV = ASSETS_DIR / "products.csv"
FEFO_CSV = ASSETS_DIR / "fefo_ranking.csv"
ALERTS_CSV = ASSETS_DIR / "model1_alerts_long.csv"
SENSOR_CSV = ASSETS_DIR / "sensor.csv"

# ---------------- App Initialization ----------------
app = FastAPI(
    title="ColdChain IQ Intelligence API",
    version="1.0.0",
    description="Production API for ColdChain IQ Monitoring, Arrhenius Decay & Early-Warning Anomaly Prediction",
)

# ---------------- CORS Configuration ----------------
# Read allowed origins from env or default to Netlify + Localhost
ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "")
if ALLOWED_ORIGINS_ENV:
    allowed_origins = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:8443",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8443",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https:\/\/.*\.netlify\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Global State & Cache ----------------
BUNDLE: Any = None
MODEL_STATUS = "NOT_INITIALIZED"
MODEL_ERROR: str | None = None
FEFO_DF: pd.DataFrame | None = None
ALERTS_DF: pd.DataFrame | None = None
BATCHES_DF: pd.DataFrame | None = None
PRODUCTS_DF: pd.DataFrame | None = None

def load_data_and_model():
    global BUNDLE, MODEL_STATUS, MODEL_ERROR, FEFO_DF, ALERTS_DF, BATCHES_DF, PRODUCTS_DF

    # 1. Load Data Files
    if FEFO_CSV.exists():
        try:
            FEFO_DF = pd.read_csv(FEFO_CSV)
            print(f"[startup] Loaded FEFO rankings: {len(FEFO_DF)} rows")
        except Exception as e:
            print(f"[startup] Error loading fefo_ranking.csv: {e}")

    if ALERTS_CSV.exists():
        try:
            ALERTS_DF = pd.read_csv(ALERTS_CSV)
            print(f"[startup] Loaded alerts: {len(ALERTS_DF)} rows")
        except Exception as e:
            print(f"[startup] Error loading model1_alerts_long.csv: {e}")

    if BATCHES_CSV.exists():
        try:
            BATCHES_DF = pd.read_csv(BATCHES_CSV)
            print(f"[startup] Loaded batches: {len(BATCHES_DF)} rows")
        except Exception as e:
            print(f"[startup] Error loading batches.csv: {e}")

    if PRODUCTS_CSV.exists():
        try:
            PRODUCTS_DF = pd.read_csv(PRODUCTS_CSV)
            print(f"[startup] Loaded products: {len(PRODUCTS_DF)} rows")
        except Exception as e:
            print(f"[startup] Error loading products.csv: {e}")

    # 2. Load Model Bundle
    if not MODEL_PATH.exists():
        MODEL_STATUS = "FILE_NOT_FOUND"
        MODEL_ERROR = f"Model artifact not found at {MODEL_PATH}"
        print(f"[startup] {MODEL_ERROR}")
        return

    try:
        from app.assets.coldchain_inference import load_bundle
        BUNDLE = load_bundle(str(MODEL_PATH))
        MODEL_STATUS = "ONLINE"
        MODEL_ERROR = None
        tasks = list(BUNDLE.get("models", {}).keys()) if isinstance(BUNDLE, dict) else []
        print(f"[startup] SUCCESS: Loaded XGBoost Early-Warning model bundle. Tasks: {tasks}")
    except Exception as e:
        MODEL_STATUS = "UNAVAILABLE_ON_HOST"
        err_str = str(e)
        if "input stream corrupted" in err_str or "XGBoostError" in err_str:
            MODEL_ERROR = (
                "XGBoost cross-platform deserialization limitation (dmlc/xgboost#12459). "
                "The model was pickled in a 64-bit Linux glibc environment and requires a Linux runtime "
                "(such as Render, Railway, or Linux Docker container) to deserialize natively."
            )
        else:
            MODEL_ERROR = f"Model load error: {err_str}"
        print(f"[startup] Model status: {MODEL_STATUS}. Detail: {MODEL_ERROR}")

@app.on_event("startup")
def startup_event():
    load_data_and_model()

# ---------------- Pydantic Request Schemas ----------------
class ReadingItem(BaseModel):
    Batch_ID: str
    Timestamp: str
    actual_temp: float | None = None
    calibrated_temp: float | None = None
    Humidity: float
    Shipment_Mode: str = "Air"

class PredictRequest(BaseModel):
    readings: list[ReadingItem]
    operating_point: str = "precision50"
    tasks: list[str] | None = None

class ArrheniusRequest(BaseModel):
    temp_celsius: float
    ea_j_per_mol: float = 51044.8

# ---------------- Endpoints ----------------
@app.get("/health")
def health_check():
    import xgboost
    return {
        "status": "ok",
        "service": "coldchain-iq-api",
        "runtime": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "xgboost_version": getattr(xgboost, "__version__", "unknown"),
        },
        "model": {
            "loaded": BUNDLE is not None,
            "artifact_name": MODEL_PATH.name,
            "status": MODEL_STATUS,
            "error_detail": MODEL_ERROR,
            "target_windows": ["t+1..t+6", "t+1..t+12", "t+1..t+24"] if BUNDLE else None,
        },
        "datasets": {
            "fefo_ranking": len(FEFO_DF) if FEFO_DF is not None else 0,
            "batches": len(BATCHES_DF) if BATCHES_DF is not None else 0,
            "alerts": len(ALERTS_DF) if ALERTS_DF is not None else 0,
            "products": len(PRODUCTS_DF) if PRODUCTS_DF is not None else 0,
            "sensor_available": SENSOR_CSV.exists(),
        },
    }

@app.get("/")
def api_root():
    return {
        "message": "ColdChain IQ Intelligence API",
        "documentation": "/docs",
        "endpoints": [
            "/health",
            "/api/fefo",
            "/api/batches",
            "/api/alerts",
            "/api/predictions/{batch_id}",
            "/api/products",
            "/api/sensor/{batch_id}",
            "/api/arrhenius",
            "/predict",
        ],
    }

@app.get("/api/fefo")
def get_fefo_rankings(
    limit: int = 50,
    offset: int = 0,
    product_id: str | None = None,
    risk: str | None = None,
):
    if FEFO_DF is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "fefo_ranking.csv not loaded")
    df = FEFO_DF
    if product_id:
        df = df[df["Product_ID"] == product_id]
    if risk:
        if risk.lower() == "high":
            df = df[(df["calibrated_temp"] > 8.0) | (df["Model1_alert_w1_6"] == True)]
        elif risk.lower() == "medium":
            df = df[(df["calibrated_temp"] >= 7.0) & (df["calibrated_temp"] <= 8.0)]
        elif risk.lower() == "low":
            df = df[(df["calibrated_temp"] < 7.0) & (df["calibrated_temp"] >= 2.0)]

    total = len(df)
    page_df = df.iloc[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "count": len(page_df),
        "rows": _to_jsonable(page_df),
    }

@app.get("/api/batches")
def get_batches(
    limit: int = 50,
    offset: int = 0,
    query: str | None = None,
):
    df = BATCHES_DF if BATCHES_DF is not None else FEFO_DF
    if df is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Batch data not available")

    if query:
        q = query.lower()
        id_col = "Batch_ID" if "Batch_ID" in df.columns else df.columns[0]
        df = df[df[id_col].astype(str).str.lower().str.contains(q)]

    total = len(df)
    page_df = df.iloc[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "count": len(page_df),
        "rows": _to_jsonable(page_df),
    }

@app.get("/api/alerts")
def get_alerts(
    limit: int = 50,
    offset: int = 0,
    only_flagged: bool = True,
):
    if ALERTS_DF is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Alerts data not available")
    df = ALERTS_DF
    if only_flagged and "alert" in df.columns:
        df = df[df["alert"] == True]

    total = len(df)
    page_df = df.iloc[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "count": len(page_df),
        "rows": _to_jsonable(page_df),
    }

@app.get("/api/predictions/{batch_id}")
def get_saved_predictions(batch_id: str):
    """Return the latest saved trained-model scores for a batch snapshot."""
    if ALERTS_DF is None or ALERTS_DF.empty:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Saved model predictions are not available",
        )

    batch_rows = ALERTS_DF.loc[ALERTS_DF["Batch_ID"] == batch_id].copy()
    if batch_rows.empty:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No saved model predictions found for {batch_id}",
        )

    batch_rows["origin_time"] = pd.to_datetime(batch_rows["origin_time"], errors="coerce")
    latest_origin = batch_rows["origin_time"].max()
    if pd.isna(latest_origin):
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Saved model predictions for {batch_id} have no valid origin time",
        )

    latest = batch_rows.loc[batch_rows["origin_time"] == latest_origin].copy()
    task_order = {"w1_6": 0, "w1_12": 1, "w1_24": 2}
    fields = [
        "Batch_ID",
        "origin_time",
        "status",
        "task",
        "window",
        "model",
        "p_cal",
        "threshold",
        "alert",
    ]
    return {
        "batch_id": batch_id,
        "inference_mode": "PRECOMPUTED_MODEL_OUTPUT",
        "origin_time": latest_origin,
        "result": _to_jsonable(
            latest[fields].sort_values(
                "task",
                key=lambda tasks: tasks.map(task_order).fillna(len(task_order)),
            )
        ),
    }

@app.get("/api/products")
def get_products():
    if PRODUCTS_DF is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Products data not available")
    return {
        "count": len(PRODUCTS_DF),
        "rows": _to_jsonable(PRODUCTS_DF),
    }

@app.get("/api/sensor/{batch_id}")
def get_sensor_readings(batch_id: str, limit: int = 240):
    """
    Returns real sensor telemetry for a given batch from sensor.csv
    """
    if not SENSOR_CSV.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sensor.csv not available")
    try:
        # Read matching rows for the batch
        # sensor.csv has: Reading_ID, Batch_ID, Timestamp, Temperature, Humidity, etc.
        chunks = []
        for chunk in pd.read_csv(SENSOR_CSV, chunksize=50000):
            matched = chunk[chunk["Batch_ID"] == batch_id]
            if len(matched) > 0:
                chunks.append(matched)
                if sum(len(c) for c in chunks) >= limit:
                    break
        if not chunks:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No readings found for {batch_id}")
        res_df = pd.concat(chunks).sort_values("Timestamp").tail(limit).copy()
        temperatures = pd.to_numeric(res_df["Temperature"], errors="coerce")
        offsets = pd.to_numeric(res_df["Calibration_Offset_Pct"], errors="coerce")
        res_df["calibrated_temp"] = temperatures * (1 + offsets / 100)
        res_df = res_df.sort_values("Timestamp").reset_index(drop=True)
        shipment_mode = None
        if BATCHES_DF is not None and {"Batch_ID", "Shipment_Mode"}.issubset(BATCHES_DF.columns):
            batch_rows = BATCHES_DF.loc[
                BATCHES_DF["Batch_ID"] == batch_id, "Shipment_Mode"
            ]
            if not batch_rows.empty:
                shipment_mode = str(batch_rows.iloc[0])
        return {
            "batch_id": batch_id,
            "shipment_mode": shipment_mode,
            "readings_count": len(res_df),
            "readings": _to_jsonable(res_df),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Error reading sensor: {e}")

@app.post("/api/arrhenius")
def calculate_arrhenius(req: ArrheniusRequest):
    """
    Calculate theoretical kinetic decay rate ratio k(T) / k_ref using Arrhenius equation:
    k(T) = A * exp(-Ea / (R * T_kelvin))
    k(T) / k_ref = exp( (-Ea / R) * (1/T - 1/T_ref) )
    """
    r_gas = 8.31446261815324  # J / (mol * K)
    t_ref = 278.15  # 5°C in Kelvin
    t_kelvin = req.temp_celsius + 273.15

    if t_kelvin <= 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Temperature in Kelvin must be > 0")

    decay_rate_ratio = math.exp((-req.ea_j_per_mol / r_gas) * (1.0 / t_kelvin - 1.0 / t_ref))
    return {
        "temp_celsius": req.temp_celsius,
        "temp_kelvin": round(t_kelvin, 2),
        "ea_j_per_mol": req.ea_j_per_mol,
        "decay_rate_ratio": round(decay_rate_ratio, 4),
        "is_accelerated": decay_rate_ratio > 1.0,
        "quality_loss_rate_per_hour": round(decay_rate_ratio / 720.0, 6),
    }

@app.post("/predict")
def run_model_inference(req: PredictRequest):
    """
    Executes real inference with the trained XGBoost Early-Warning model bundle.
    If the model bundle could not be deserialized on the host environment (e.g. Windows MSVC),
    this endpoint clearly returns an error and identifies the ML model as unavailable.
    It NEVER fabricates synthetic predictions as genuine model output.
    """
    if BUNDLE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "model_available": False,
                "error": "Trained XGBoost model artifact is not loaded in this environment.",
                "reason": MODEL_ERROR,
                "recommended_action": "Deploy the backend to a Linux environment (Docker / Render / Railway) to run native inference.",
            },
        )

    try:
        from app.assets.coldchain_inference import prepare, predict

        raw_records = [r.dict() for r in req.readings]
        df = pd.DataFrame(raw_records)

        # Ensure calibrated_temp or actual_temp column exists
        if "actual_temp" not in df.columns or df["actual_temp"].isna().all():
            if "calibrated_temp" in df.columns:
                df["actual_temp"] = df["calibrated_temp"]

        history = prepare(df)
        kwargs = {"operating_point": req.operating_point}
        if req.tasks is not None:
            kwargs["tasks"] = req.tasks

        # Genuine XGBoost inference
        result = predict(history, BUNDLE, **kwargs)
        return {
            "success": True,
            "model_version": BUNDLE.get("version", "FINAL-1.0"),
            "operating_point": req.operating_point,
            "inference_mode": "GENUINE_XGBOOST_EARLY_WARNING",
            "result": _to_jsonable(result),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {e}",
        )

# ---------------- Utility Serializer ----------------
def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return obj.replace({np.nan: None}).reset_index(drop=True).to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.replace({np.nan: None}).to_dict()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj

# ---------------- Direct Entrypoint ----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting ColdChain IQ API on {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
