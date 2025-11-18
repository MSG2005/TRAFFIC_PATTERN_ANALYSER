import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Make sure the 'data' folder exists
os.makedirs("data", exist_ok=True)

# ---------- CONFIG ----------
START_DATE = datetime(2024, 1, 1)  # start date for our synthetic data
NUM_DAYS = 30                      # number of days to simulate

ROADS = [
    {"road_name": "NH-16 Main", "location_id": 1},
    {"road_name": "City Ring Road", "location_id": 2},
    {"road_name": "IT Park Road", "location_id": 3},
]

# ---------- DATA GENERATION ----------
rows = []
id_counter = 1

for day in range(NUM_DAYS):
    for hour in range(24):
        ts = START_DATE + timedelta(days=day, hours=hour)
        day_of_week = ts.weekday()         # 0 = Monday, 6 = Sunday
        is_weekend = 1 if day_of_week >= 5 else 0
        is_holiday = 0                     # keep it simple for now

        for road in ROADS:
            # Base vehicle count pattern by hour
            # Night (0-5)      -> very low
            # Morning peak     -> high (8-11)
            # Daytime          -> medium (11-16)
            # Evening peak     -> high (17-20)
            # Late evening     -> medium-low (21-23)
            if 8 <= hour <= 11 or 17 <= hour <= 20:
                base = 80     # peak
            elif 7 <= hour < 8 or 11 < hour <= 16 or 20 < hour <= 22:
                base = 50     # medium
            elif 0 <= hour <= 5:
                base = 10     # very low
            else:
                base = 30     # normal

            # Weekends usually slightly lower traffic
            if is_weekend:
                base *= 0.7

            # IT Park Road: more traffic during office times
            if road["road_name"] == "IT Park Road":
                if 9 <= hour <= 12 or 18 <= hour <= 21:
                    base *= 1.3

            # Add some randomness around the base
            vehicle_count = max(0, int(np.random.normal(loc=base, scale=8)))

            # Average speed decreases with more traffic (simple relation)
            avg_speed = 40 - (vehicle_count / 10)   # base formula
            avg_speed = np.random.normal(loc=avg_speed, scale=5)
            avg_speed = float(np.clip(avg_speed, 10, 60))  # keep in [10, 60]

            rows.append({
                "id": id_counter,
                "timestamp": ts,
                "road_name": road["road_name"],
                "location_id": road["location_id"],
                "vehicle_count": vehicle_count,
                "avg_speed": round(avg_speed, 2),
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday
            })

            id_counter += 1

# ---------- SAVE TO CSV ----------
df = pd.DataFrame(rows)

output_path = os.path.join("data", "traffic_data.csv")
df.to_csv(output_path, index=False)

print(f"Data generated successfully! Rows: {len(df)}")
print(f"Saved to: {output_path}")
print("\nSample:")
print(df.head())
