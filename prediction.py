import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

print("\n===================================")
print(" LOCATION PREDICTION MODEL")
print("===================================")

# Load analyzed dataset
df = pd.read_csv("data/movement_analysis.csv")

print("\nDataset loaded!")
print("Total records:", len(df))


# =========================================
# PREPARE TARGET
# =========================================

# Area is our target
target = "Area"

# Convert Area names into numbers
label_encoder = LabelEncoder()

df["Area_Encoded"] = label_encoder.fit_transform(df[target])


# =========================================
# FEATURES
# =========================================

features = [
    "Person_ID",
    "Latitude",
    "Longitude",
    "Hour",
    "Speed",
    "Distance",
    "Movement_Cluster"
]

X = df[features]
y = df["Area_Encoded"]


# =========================================
# TRAIN / TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================
# RANDOM FOREST
# =========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=10
)

model.fit(X_train, y_train)

print("\nRandom Forest training completed!")


# =========================================
# EVALUATION
# =========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print(" MODEL PERFORMANCE")
print("===================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# =========================================
# FEATURE IMPORTANCE
# =========================================

print("\n===================================")
print(" FEATURE IMPORTANCE")
print("===================================")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))


# =========================================
# SAMPLE PREDICTION
# =========================================

print("\n===================================")
print(" SAMPLE LOCATION PREDICTION")
print("===================================")

sample = pd.DataFrame([{
    "Person_ID": 1,
    "Latitude": 23.2500,
    "Longitude": 77.4300,
    "Hour": 18,
    "Speed": 30,
    "Distance": 8,
    "Movement_Cluster": 1
}])

prediction = model.predict(sample)[0]

probabilities = model.predict_proba(sample)[0]

predicted_area = label_encoder.inverse_transform(
    [prediction]
)[0]

print("\nPredicted Area:", predicted_area)

print("\nLocation probabilities:")

for area, probability in zip(
    label_encoder.classes_,
    probabilities
):
    print(f"{area}: {probability * 100:.2f}%")


# =========================================
# SAVE MODEL
# =========================================

joblib.dump(
    model,
    "models/location_prediction_model.pkl"
)

joblib.dump(
    label_encoder,
    "models/area_label_encoder.pkl"
)

print("\n===================================")
print(" MODEL SAVED")
print("===================================")

print("\nSaved:")
print("models/location_prediction_model.pkl")
print("models/area_label_encoder.pkl")