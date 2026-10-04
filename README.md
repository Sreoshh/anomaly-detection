# Automatic Weather Station Anomaly Detection

A time-series machine learning project that detects unusual weather observations using statistical baselines, Isolation Forest, and an Autoencoder. It combines anomaly scores and severity labels and provides a FastAPI backend for accessing results.

## Features

- Weather data preprocessing and feature engineering
- Statistical anomaly detection baselines
- Isolation Forest and Autoencoder models
- Combined anomaly scoring and severity classification
- REST API using FastAPI
- React dashboard for visualising anomaly results

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, PyTorch, FastAPI, Uvicorn, React, and Vite.

## Prerequisites

- Python installed
- Node.js and npm installed
- Git installed

## Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Sreoshh/anomaly-detection.git
cd anomaly-detection
```

### 1. Install backend dependencies

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare the data

Place the raw weather dataset at:

`data/raw/abohar_weather.csv`

Ensure the processed results file exists at:

`data/processed/weather_anomaly_scores.csv`

If the processed file is missing, run the project's ML pipeline to generate it before starting the API.

### 3. Start the backend

From the project root, with the virtual environment activated:

```powershell
python -m uvicorn api.main:app --reload
```

Backend: `http://127.0.0.1:8000`

### 4. Start the frontend

Open a **second terminal**:

```powershell
cd dashboard
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

## Verify the Application

With both servers running, check:

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8000/health` | Check API status and results file availability |
| `http://127.0.0.1:8000/summary` | View observation and anomaly statistics |
| `http://127.0.0.1:8000/anomalies` | Retrieve anomaly results |
| `http://127.0.0.1:8000/docs` | Explore API endpoints interactively |
| Frontend URL from Vite | Open the dashboard |

## Notes

- Keep the backend and frontend running in separate terminals.
- The frontend requires the backend to be accessible to load API data.
- If the API reports that the results CSV is missing, generate the processed results using the ML pipeline.
- Detected anomalies are model generated candidates and are not necessarily confirmed weather events.

## License

