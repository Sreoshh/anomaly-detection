import pandas as pd


def load_weather_data(file_path):
    """Load and prepare the raw weather dataset."""

    df = pd.read_csv(file_path, skiprows=2)

    df["time"] = pd.to_datetime(df["time"])

    df = df.rename(columns={
        "time": "timestamp",
        "temperature_2m (°C)": "temperature",
        "relative_humidity_2m (%)": "humidity",
        "pressure_msl (hPa)": "pressure"
    })

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def add_time_features(df):
    """Add basic time-based features."""

    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year

    return df