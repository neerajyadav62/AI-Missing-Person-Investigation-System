#  CASEFILE AI
## AI-Powered Missing Person Investigation and Location Prediction System

###  Project Overview

CASEFILE AI is an academic machine learning project designed to analyze movement patterns and provide probabilistic predictions of probable locations.

The system combines movement analysis, anomaly detection, supervised machine learning, route prediction, search-priority scoring, and interactive visualization in a Streamlit dashboard.

>  This project is an academic simulation using synthetic fictional case information. Predictions are probabilistic and should not be treated as proof of a person's real-world location.

---

## 🎯 Objectives

- Analyze historical movement data
- Identify common movement patterns
- Detect unusual movement behavior
- Predict probable locations
- Predict possible movement routes
- Generate search-priority areas
- Provide explanations for predictions
- Visualize movement data on an interactive map

---

## 🧠 Machine Learning Techniques

### 1. K-Means Clustering
Used to identify groups of similar movement locations and patterns.

### 2. Isolation Forest
Used for detecting anomalous movement observations.

### 3. Random Forest
Used for predicting the probable movement area.

### 4. Markov Chain
Used to analyze area-to-area transitions and predict possible next areas.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Folium
- Matplotlib
- Plotly
- Joblib

---

## 📊 Project Results

The current academic prototype contains:

- 315 movement records
- 5 movement clusters
- 16 detected anomalies
- 88.89% test accuracy on the synthetic dataset

---

## 🗺️ Dashboard Features

The Streamlit dashboard provides:

- Probable Location Prediction
- Search Priority Areas
- Route Prediction
- Detected Anomalies
- Interactive Movement Map
- AI Prediction Explanation

---

## 📁 Project Structure

```text
CASEFILE AI
│
├── app.py
├── main.py
├── analysis.py
├── prediction.py
├── route_priority.py
│
├── data/
│   ├── movement_data.csv
│   ├── movement_analysis.csv
│   ├── search_priority.csv
│   └── route_transition_matrix.csv
│
└── models/
    ├── location_prediction_model.pkl
    └── area_label_encoder.pkl
