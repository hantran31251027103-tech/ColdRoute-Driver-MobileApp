# ColdChain IQ — Backend Deployment Guide

This backend provides FastAPI endpoints, the Arrhenius degradation engine, and the XGBoost Early-Warning model (`FINAL-1.0`).

## 1. Hosting Recommendation

**Recommended Platform:** [Render](https://render.com) (Web Service) or [Railway](https://railway.app) / [Fly.io](https://fly.io) using Linux Docker.

### Why Linux Hosting is Required for the Model Artifact:
- The trained XGBoost model bundle (`coldchain_early_warning_FINAL.pkl`) was serialized in an x86_64 Linux glibc environment (Google Colab, Python 3.12.3, `xgboost==3.4.1`).
- As documented in **[dmlc/xgboost Issue #12459](https://github.com/dmlc/xgboost/issues/12459)**, XGBoost's C++ buffer serializer produces `input stream corrupted` when attempting cross-platform unpickling on Windows MSVC runtimes due to standard library stream extraction differences.
- On standard Linux cloud containers (Render, Railway, Fly.io, Ubuntu Docker), the artifact deserializes and executes native inference seamlessly.

---

## 2. Option A: Deploy to Render (Easiest)

The repository-root `render.yaml` is a Render Blueprint that configures the Linux
Python service, points it at `Backend`, and sets the health check and runtime.

1. Push this repository to GitHub. The trained model artifact and required CSV
   assets must be committed; `features_engineered.csv` is intentionally excluded.
2. Log in to [Render Dashboard](https://dashboard.render.com), choose **New +**
   -> **Blueprint**, and connect this repository.
3. Review the `coldchain-iq-api` service from `render.yaml` and apply it.
4. Wait for the first deploy to finish, then open:
   `https://coldchain-iq-api.onrender.com/health`
5. Confirm `runtime.os` is `Linux` and `model.loaded` is `true`. Do not rely on
   model risk output if the health response reports the model as unavailable.

The backend permits Netlify origins through its CORS origin pattern, and the
Blueprint explicitly allows `localhost:5173` and `127.0.0.1:5173` for local UI
development. A custom frontend domain must be added to `ALLOWED_ORIGINS`.

After deployment, configure the frontend's `VITE_API_URL` to the Render service
origin (for example, `https://coldchain-iq-api.onrender.com`, without a trailing
slash). Set it in Netlify and redeploy, or put it in the mobile app's
`.env.local` for local development.

---

## 3. Option B: Deploy with Docker (Render / Railway / Fly.io / VPS)

A production-ready `Dockerfile` based on `python:3.12-slim` is provided.

```bash
# Build the Docker image
docker build -t coldchain-iq-backend -f Dockerfile .

# Run container locally
docker run -p 8000:8000 -e PORT=8000 coldchain-iq-backend
```

Test health:
```bash
curl http://localhost:8000/health
```

---

## 4. Verification Endpoints

- Health Check: `GET /health`
- FEFO Rankings: `GET /api/fefo?limit=20`
- Alerts: `GET /api/alerts?limit=10`
- Batch Telemetry: `GET /api/sensor/BATCH-01004`
- Arrhenius Engine: `POST /api/arrhenius` with `{"temp_celsius": 8.5}`
- Early-Warning Inference: `POST /predict` with batch history
