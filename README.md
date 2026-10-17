# ColdRoute Driver mobile app

The responsive React app reads FEFO shipment data and telemetry from the ColdChain FastAPI backend. Shipment risk is produced by the trained XGBoost model through `/predict`; model failures are shown in the app rather than replaced with simulated predictions.

## Run locally

Use two terminals. The frontend's Vite proxy forwards `/api`, `/health`, and `/predict` to `http://127.0.0.1:8000` by default.

Terminal 1 — start the backend from the repository root:

```powershell
Set-Location Backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 — start the mobile web app:

```powershell
Set-Location "UI\ColdRoute Driver Mobile App"
pnpm install --frozen-lockfile
pnpm dev
```

Open `http://localhost:5173`. The demo sign-in is `DRV-8492` / `coldroute`; it is a frontend demo login, not backend authentication. To use a backend on another host, set `API_PROXY_TARGET` before starting Vite, or set `VITE_API_URL` to the backend origin.

The model artifact requires a compatible Linux runtime. On Windows, the API automatically displays the saved trained-model scores for the exact shipment snapshot when live XGBoost inference is unavailable. These historical scores are clearly labelled and are not live predictions.

To run the UI locally while using the live model on the deployed Linux backend, create `.env.local` in this app directory with the Render API URL:

```dotenv
VITE_API_URL=https://coldchain-iq-api.onrender.com
```

Then start Vite as usual. The API already allows `localhost:5173` and `127.0.0.1:5173` as origins. For live local inference without cloud, run the backend in WSL2 or Docker:

```powershell
Set-Location Backend
docker build -t coldchain-iq-backend .
docker run --rm -p 8000:8000 coldchain-iq-backend
```

## Deploy

Netlify builds the frontend from `UI/ColdRoute Driver Mobile App` using the repository's root `netlify.toml`. Deploy the trained model in the separate Linux FastAPI service using the repository-root `render.yaml`; see [Backend/README_DEPLOY.md](../../Backend/README_DEPLOY.md) for the steps.

In Netlify, set `VITE_API_URL` to the deployed backend origin (for example, `https://your-api.onrender.com`, without a trailing slash), then trigger a new deploy so Vite embeds it in the frontend. The backend CORS configuration allows Netlify sites and local development origins. Confirm `/health` reports `"model": {"loaded": true}` before relying on model risk output.

If you use a custom frontend domain, add that exact origin to the backend's `ALLOWED_ORIGINS` environment variable.

## Data and model endpoints used

- `GET /health` — API and model readiness
- `GET /api/fefo?limit=20` — FEFO queue
- `GET /api/sensor/{batch_id}?limit=240` — latest calibrated sensor history
- `POST /predict` — trained-model risk predictions for 6, 12, and 24 hours
- `GET /api/predictions/{batch_id}` — saved model scores for the matching snapshot when live inference is unavailable
