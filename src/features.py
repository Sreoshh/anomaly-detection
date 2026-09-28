def add_weather_features(df):
    """Add change, rolling, and deviation features."""

    df = df.copy()

    # Hourly changes
    df["temperature_change"] = df["temperature"].diff()
    df["humidity_change"] = df["humidity"].diff()
    df["pressure_change"] = df["pressure"].diff()

    # 24-hour rolling statistics
    df["temperature_rolling_mean"] = df["temperature"].rolling(24).mean()
    df["temperature_rolling_std"] = df["temperature"].rolling(24).std()

    df["humidity_rolling_mean"] = df["humidity"].rolling(24).mean()
    df["humidity_rolling_std"] = df["humidity"].rolling(24).std()

    df["pressure_rolling_mean"] = df["pressure"].rolling(24).mean()
    df["pressure_rolling_std"] = df["pressure"].rolling(24).std()

    # Deviation from rolling mean
    df["temperature_deviation"] = (
        df["temperature"] - df["temperature_rolling_mean"]
    )

    df["humidity_deviation"] = (
        df["humidity"] - df["humidity_rolling_mean"]
    )

    df["pressure_deviation"] = (
        df["pressure"] - df["pressure_rolling_mean"]
    )

    return df