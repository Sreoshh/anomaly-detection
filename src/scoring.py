def normalize_isolation_score(anomaly_score):
    """Convert Isolation Forest scores to a 0–1 anomaly score."""

    return 1 - anomaly_score.rank(pct=True)


def normalize_autoencoder_score(reconstruction_error):
    """Convert reconstruction errors to a 0–1 anomaly score."""

    return reconstruction_error.rank(pct=True)


def calculate_combined_score(
    isolation_score,
    autoencoder_score,
    isolation_weight=0.5,
    autoencoder_weight=0.5
):
    """Combine normalized model scores."""

    return (
        isolation_weight * isolation_score
        + autoencoder_weight * autoencoder_score
    )


def classify_severity(score):
    """Classify anomaly severity."""

    if score >= 0.90:
        return "High"
    elif score >= 0.75:
        return "Medium"

    return "Low"