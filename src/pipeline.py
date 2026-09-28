
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import load_weather_data, add_time_features
from src.features import add_weather_features
from src.anomaly_models import (
    train_isolation_forest,
    get_isolation_predictions,
    Autoencoder,
)
from src.scoring import (
    normalize_isolation_score,
    normalize_autoencoder_score,
    calculate_combined_score,
    classify_severity,
)


FEATURES = [
    "temperature", "humidity", "pressure",
    "hour", "month",
    "temperature_change", "humidity_change", "pressure_change",
    "temperature_rolling_mean", "temperature_rolling_std",
    "humidity_rolling_mean", "humidity_rolling_std",
    "pressure_rolling_mean", "pressure_rolling_std",
    "temperature_deviation", "humidity_deviation",
    "pressure_deviation",
]


def run_pipeline(data_path, epochs=30, batch_size=128):
    """Train both models and return anomaly results for the test period."""

    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load data and engineer features
    df = load_weather_data(data_path)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    # 2. Chronological 70/15/15 split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    X_train = train_df[FEATURES]
    X_val = val_df[FEATURES]
    X_test = test_df[FEATURES]

    # 3. Train Isolation Forest on training data only
    isolation_model = train_isolation_forest(
        X_train,
        contamination=0.1,
        random_state=42,
    )

    isolation_predictions, isolation_raw_scores = (
        get_isolation_predictions(isolation_model, X_test)
    )

    # 4. Scale Autoencoder inputs using training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    # 5. Train Autoencoder
    model = Autoencoder(input_dim=len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()

        for (batch,) in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

    # 6. Set anomaly threshold using validation reconstruction errors
    model.eval()

    with torch.no_grad():
        val_reconstructed = model(val_tensor)
        val_errors = torch.mean(
            (val_reconstructed - val_tensor) ** 2, dim=1
        ).numpy()

        test_reconstructed = model(test_tensor)
        test_errors = torch.mean(
            (test_reconstructed - test_tensor) ** 2, dim=1
        ).numpy()

    threshold = float(np.percentile(val_errors, 95))
    autoencoder_anomalies = test_errors > threshold

    # 7. Build unified test results
    results = test_df[
        ["timestamp", "temperature", "humidity", "pressure"]
    ].copy()

    results["isolation_anomaly"] = isolation_predictions == -1
    results["autoencoder_anomaly"] = autoencoder_anomalies
    results["anomaly_score"] = isolation_raw_scores
    results["reconstruction_error"] = test_errors

    results["isolation_score"] = normalize_isolation_score(
        results["anomaly_score"]
    )

    results["autoencoder_score"] = normalize_autoencoder_score(
        results["reconstruction_error"]
    )

    results["combined_anomaly_score"] = calculate_combined_score(
        results["isolation_score"],
        results["autoencoder_score"],
    )

    results["severity"] = results["combined_anomaly_score"].apply(
        classify_severity
    )

    results["final_anomaly"] = (
        results["isolation_anomaly"]
        | results["autoencoder_anomaly"]
    )

    return results, model, isolation_model, scaler, threshold
