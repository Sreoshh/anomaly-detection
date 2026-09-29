
from pathlib import Path

import json

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "weather_anomaly_scores.csv"
)

app = FastAPI(
    title="Weather Anomaly Detection API",
    description="API for weather observations and detected anomalies.",
    version="0.1.0",
)

# Allow the local React dashboard to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def load_results() -> pd.DataFrame:
    """Load the saved model results."""
    if not RESULTS_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Results CSV not found. Run the ML pipeline first.",
        )

    try:
        df = pd.read_csv(RESULTS_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
        return df
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read the saved results file.",
        ) from exc


@app.get("/")
def root():
    return {
        "message": "Weather Anomaly Detection API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "results_file_exists": RESULTS_PATH.is_file(),
    }


@app.get("/summary")
def summary():
    """Return aggregate statistics for the saved test-period results."""
    df = load_results()

    anomaly_count = int(df["final_anomaly"].astype(bool).sum())

    return {
        "total_observations": len(df),
        "total_anomalies": anomaly_count,
        "normal_observations": len(df) - anomaly_count,
        "anomaly_percentage": round(
            anomaly_count / len(df) * 100, 2
        ) if len(df) else 0,
        "severity_distribution": {
            str(key): int(value)
            for key, value in df["severity"].value_counts().items()
        },
        "period_start": df["timestamp"].min().isoformat() if len(df) else None,
        "period_end": df["timestamp"].max().isoformat() if len(df) else None,
    }


@app.get("/anomalies")
def anomalies(
    anomaly_only: bool = False,
    severity: str | None = Query(
        default=None,
        description="Filter by Low, Medium, or High severity.",
    ),
    start: str | None = Query(default=None, description="Start timestamp"),
    end: str | None = Query(default=None, description="End timestamp"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """Return paginated weather results with optional filters."""
    df = load_results()

    if severity is not None:
        severity = severity.strip().title()
        if severity not in {"Low", "Medium", "High"}:
            raise HTTPException(
                status_code=422,
                detail="severity must be Low, Medium, or High.",
            )
        df = df[df["severity"] == severity]

    if anomaly_only:
        df = df[df["final_anomaly"].astype(bool)]

    try:
        if start is not None:
            start_time = pd.to_datetime(start, errors="raise")
            df = df[df["timestamp"] >= start_time]

        if end is not None:
            end_time = pd.to_datetime(end, errors="raise")
            df = df[df["timestamp"] <= end_time]
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid start or end timestamp.",
        ) from exc

    if start is not None and end is not None and start_time > end_time:
        raise HTTPException(
            status_code=422,
            detail="start must be earlier than or equal to end.",
        )

    total = len(df)
    page = df.iloc[offset:offset + limit]

    # Produce JSON-safe values, including timestamps and missing values.
    records = json.loads(
        page.to_json(orient="records", date_format="iso")
    )

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "returned": len(records),
        "results": records,
    }
