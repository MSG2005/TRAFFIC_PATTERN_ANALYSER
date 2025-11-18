from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
import os

# ---------------- CONFIG & INITIALIZATION ----------------

app = FastAPI(
    title="Traffic Pattern Analyzer API",
    version="1.0.0",
    description="Backend API for Traffic Pattern Analyzer project",
)

# Allow frontend (browser) to call this API from any origin (for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "traffic_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_model.pkl")

# Load dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"traffic_data.csv not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# Basic safety checks
required_columns = [
    "timestamp", "road_name", "location_id", "vehicle_count",
    "avg_speed", "day_of_week", "is_weekend", "is_holiday"
]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in traffic_data.csv: {missing_cols}")

# Sort for time-based operations
df = df.sort_values(["road_name", "timestamp"]).reset_index(drop=True)

# Load model artifact
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model_artifact = joblib.load(MODEL_PATH)
model = model_artifact["model"]
feature_cols = model_artifact["feature_cols"]  # ["hour", "day_of_week", "is_weekend", "location_id", "rolling_mean_3h"]

# ---------------- Pydantic Models ----------------

class PredictRequest(BaseModel):
    road_name: str
    timestamp: datetime  # ISO string like "2024-01-05T10:00:00"


class PredictResponse(BaseModel):
    predicted_vehicle_count: float
    congestion_level: str
    used_features: dict


# ---------------- Helper Functions ----------------

def get_rolling_mean_3h(road_name: str, ts: datetime) -> float:
    """
    Compute rolling mean of vehicle_count for the last 3 records for this road
    before the given timestamp.
    If not enough data, fallback to overall mean for this road.
    """
    road_data = df[df["road_name"] == road_name]
    past_data = road_data[road_data["timestamp"] < ts].sort_values("timestamp", ascending=False)

    if len(past_data) == 0:
        # fallback: mean for that road across all time
        fallback_mean = float(road_data["vehicle_count"].mean())
        return fallback_mean

    last_3 = past_data.head(3)
    return float(last_3["vehicle_count"].mean())


def compute_congestion_level(predicted_count: float) -> str:
    """
    Simple rule-based congestion level.
    You can tune the thresholds based on your data distribution.
    """
    if predicted_count >= 80:
        return "High"
    elif predicted_count >= 40:
        return "Medium"
    else:
        return "Low"


# ---------------- API Endpoints ----------------

@app.get("/api/roads")
def get_roads():
    """
    Returns a list of available roads in the dataset.
    Useful for frontend dropdowns.
    """
    roads = (
        df[["road_name", "location_id"]]
        .drop_duplicates()
        .sort_values("location_id")
        .to_dict(orient="records")
    )
    return {"roads": roads}


@app.get("/api/traffic")
def get_traffic(road_name: str, date: str):
    """
    Get traffic data for a given road and date (YYYY-MM-DD).
    Returns a list of hourly records.
    """
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    road_mask = df["road_name"] == road_name
    date_mask = df["timestamp"].dt.date == target_date
    filtered = df[road_mask & date_mask].copy()

    if filtered.empty:
        raise HTTPException(status_code=404, detail="No data found for this road and date.")

    filtered = filtered.sort_values("timestamp")

    # Format timestamps as ISO strings for JSON
    filtered["timestamp"] = filtered["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Only send relevant columns
    result = filtered[[
        "timestamp",
        "road_name",
        "location_id",
        "vehicle_count",
        "avg_speed",
        "day_of_week",
        "is_weekend",
        "is_holiday"
    ]].to_dict(orient="records")

    return {"data": result}


@app.post("/api/predict", response_model=PredictResponse)
def predict_traffic(req: PredictRequest):
    """
    Predict vehicle_count and congestion level for the given road and timestamp.
    Uses the trained RandomForest model.
    """
    # Validate road_name
    road_rows = df[df["road_name"] == req.road_name]
    if road_rows.empty:
        raise HTTPException(status_code=404, detail="Unknown road_name. Check /api/roads for valid options.")

    # Extract basic features
    ts = req.timestamp
    hour = ts.hour
    day_of_week = ts.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    location_id = int(road_rows["location_id"].iloc[0])

    # Compute rolling_mean_3h using past data
    rolling_mean_3h = get_rolling_mean_3h(req.road_name, ts)

    # Build feature dict
    feature_values = {
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "location_id": location_id,
        "rolling_mean_3h": rolling_mean_3h,
    }

    # Order features according to training feature_cols
    X = [[feature_values[col] for col in feature_cols]]

    # Predict
    predicted = float(model.predict(X)[0])
    congestion_level = compute_congestion_level(predicted)

    return PredictResponse(
        predicted_vehicle_count=predicted,
        congestion_level=congestion_level,
        used_features=feature_values,
    )


@app.get("/api/anomalies")
def get_anomalies(road_name: str, date: str):
    """
    Simple anomaly detection:
    For the specified road and date, marks hours where vehicle_count
    is significantly higher than the day's mean (mean + 2*std deviation).
    """
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    road_mask = df["road_name"] == road_name
    date_mask = df["timestamp"].dt.date == target_date
    day_data = df[road_mask & date_mask].copy()

    if day_data.empty:
        raise HTTPException(status_code=404, detail="No data found for this road and date.")

    mean_count = day_data["vehicle_count"].mean()
    std_count = day_data["vehicle_count"].std(ddof=0)  # population std

    if np.isnan(std_count) or std_count == 0:
        # Not enough variation to detect anomalies
        return {"anomalies": [], "mean": mean_count, "std": float(std_count)}

    threshold = mean_count + 2 * std_count

    anomalies = day_data[day_data["vehicle_count"] > threshold].copy()
    anomalies = anomalies.sort_values("timestamp")

    anomalies["timestamp"] = anomalies["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    result = anomalies[[
        "timestamp",
        "road_name",
        "location_id",
        "vehicle_count",
        "avg_speed"
    ]].to_dict(orient="records")

    return {
        "mean": mean_count,
        "std": float(std_count),
        "threshold": threshold,
        "anomalies": result,
    }


@app.get("/")
def root():
    return {"message": "Traffic Pattern Analyzer API is running. Visit /docs for interactive docs."}
