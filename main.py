import pandas as pd
import numpy as np

# -------------------------------
# CASEFILE - Synthetic GPS Dataset
# -------------------------------

np.random.seed(42)

# Locations used in our fictional case
locations = {
    "Home": (23.2599, 77.4126),
    "College": (23.2156, 77.4320),
    "Market": (23.2500, 77.4300),
    "Park": (23.2700, 77.4000),
    "Railway Station": (23.2599, 77.4120),
    "Unknown Area": (23.3000, 77.4800)
}

rows = []

# Generate normal historical movement
for person in range(1, 11):

    for i in range(30):

        # Mostly normal locations
        area = np.random.choice(
            ["Home", "College", "Market", "Park", "Railway Station"],
            p=[0.35, 0.25, 0.20, 0.10, 0.10]
        )

        lat, lon = locations[area]

        # Small GPS variation
        lat += np.random.normal(0, 0.002)
        lon += np.random.normal(0, 0.002)

        hour = np.random.choice(
            [7, 8, 9, 10, 13, 17, 18, 19, 20, 21]
        )

        speed = np.random.uniform(15, 45)
        distance = np.random.uniform(1, 12)

        rows.append([
            person,
            lat,
            lon,
            hour,
            speed,
            distance,
            area
        ])

# Add unusual/anomalous movements
for i in range(15):

    lat, lon = locations["Unknown Area"]

    lat += np.random.normal(0, 0.002)
    lon += np.random.normal(0, 0.002)

    rows.append([
        np.random.randint(1, 11),
        lat,
        lon,
        np.random.choice([1, 2, 3, 23]),
        np.random.uniform(60, 100),
        np.random.uniform(25, 50),
        "Unknown Area"
    ])

# Create dataframe
df = pd.DataFrame(
    rows,
    columns=[
        "Person_ID",
        "Latitude",
        "Longitude",
        "Hour",
        "Speed",
        "Distance",
        "Area"
    ]
)

# Save dataset
df.to_csv("data/movement_data.csv", index=False)

print("\n===================================")
print(" CASEFILE DATASET CREATED")
print("===================================")

print(f"Total records: {len(df)}")

print("\nFirst 5 records:")
print(df.head())

print("\nArea distribution:")
print(df["Area"].value_counts())

print("\nDataset saved to:")
print("data/movement_data.csv")