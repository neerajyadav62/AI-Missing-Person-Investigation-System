import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("\n===================================")
print(" MOVEMENT ANALYSIS STARTED")
print("===================================")

# Load dataset
df = pd.read_csv("data/movement_data.csv")

print("\nDataset loaded successfully!")
print("Total records:", len(df))

# =========================================================
# 1. K-MEANS CLUSTERING
# =========================================================

print("\n--- K-MEANS MOVEMENT CLUSTERING ---")

features_cluster = [
    "Latitude",
    "Longitude",
    "Hour",
    "Speed",
    "Distance"
]

X_cluster = df[features_cluster].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Create 5 movement clusters
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Movement_Cluster"] = kmeans.fit_predict(X_scaled)

print("K-Means clustering completed.")

print("\nCluster distribution:")
print(df["Movement_Cluster"].value_counts().sort_index())


# =========================================================
# 2. ISOLATION FOREST - ANOMALY DETECTION
# =========================================================

print("\n--- ISOLATION FOREST ANOMALY DETECTION ---")

features_anomaly = [
    "Latitude",
    "Longitude",
    "Hour",
    "Speed",
    "Distance"
]

X_anomaly = df[features_anomaly]

isolation_forest = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["Anomaly_Result"] = isolation_forest.fit_predict(X_anomaly)

# Convert:
# -1 = Anomaly
#  1 = Normal

df["Anomaly"] = df["Anomaly_Result"].apply(
    lambda x: "Anomaly" if x == -1 else "Normal"
)

print("Isolation Forest completed.")

print("\nAnomaly summary:")
print(df["Anomaly"].value_counts())


# =========================================================
# 3. SHOW ANOMALOUS LOCATIONS
# =========================================================

anomalies = df[df["Anomaly"] == "Anomaly"]

print("\nNumber of detected anomalies:", len(anomalies))

print("\nSample anomalous records:")

print(
    anomalies[
        [
            "Person_ID",
            "Latitude",
            "Longitude",
            "Hour",
            "Speed",
            "Distance",
            "Area",
            "Anomaly"
        ]
    ].head(10)
)


# =========================================================
# 4. SAVE ANALYSIS DATA
# =========================================================

df.to_csv("data/movement_analysis.csv", index=False)

print("\n===================================")
print(" MOVEMENT ANALYSIS COMPLETED")
print("===================================")

print("\nAnalysis file saved to:")
print("data/movement_analysis.csv")