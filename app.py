import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium

from streamlit_folium import st_folium


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CASEFILE AI",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🔎 CASEFILE AI")
st.subheader(
    "AI-Powered Missing Person Investigation "
    "and Location Prediction System"
)

st.info(
    "Academic simulation using synthetic movement data. "
    "Predictions are probabilistic and should not be treated "
    "as proof of a person's location."
)


# =========================================================
# LOAD DATA
# =========================================================

movement = pd.read_csv(
    "data/movement_analysis.csv"
)

priority = pd.read_csv(
    "data/search_priority.csv"
)

route = pd.read_csv(
    "data/route_transition_matrix.csv",
    index_col=0
)

model = joblib.load(
    "models/location_prediction_model.pkl"
)

label_encoder = joblib.load(
    "models/area_label_encoder.pkl"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Case Information")

person_id = st.sidebar.selectbox(
    "Select Person",
    sorted(movement["Person_ID"].unique())
)

person_data = movement[
    movement["Person_ID"] == person_id
]

last_record = person_data.iloc[-1]

st.sidebar.write(
    f"**Person ID:** {person_id}"
)

st.sidebar.write(
    f"**Last Area:** {last_record['Area']}"
)

st.sidebar.write(
    f"**Last Hour:** {last_record['Hour']}:00"
)


# =========================================================
# TOP METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        len(movement)
    )

with col2:
    anomaly_count = (
        movement["Anomaly"] == "Anomaly"
    ).sum()

    st.metric(
        "Anomalies",
        anomaly_count
    )

with col3:
    st.metric(
        "Movement Clusters",
        movement["Movement_Cluster"].nunique()
    )

with col4:
    st.metric(
        "Model Accuracy",
        "88.89%"
    )


# =========================================================
# LOCATION PREDICTION
# =========================================================

st.header("📍 Probable Location Prediction")

features = [
    "Person_ID",
    "Latitude",
    "Longitude",
    "Hour",
    "Speed",
    "Distance",
    "Movement_Cluster"
]

sample = pd.DataFrame([{
    "Person_ID": person_id,
    "Latitude": last_record["Latitude"],
    "Longitude": last_record["Longitude"],
    "Hour": last_record["Hour"],
    "Speed": last_record["Speed"],
    "Distance": last_record["Distance"],
    "Movement_Cluster": last_record["Movement_Cluster"]
}])

probabilities = model.predict_proba(sample)[0]

prediction_table = pd.DataFrame({
    "Area": label_encoder.classes_,
    "Probability (%)": probabilities * 100
})

prediction_table = prediction_table.sort_values(
    "Probability (%)",
    ascending=False
)

prediction_table["Probability (%)"] = (
    prediction_table["Probability (%)"].round(2)
)

st.dataframe(
    prediction_table,
    use_container_width=True,
    hide_index=True
)

predicted_area = prediction_table.iloc[0]["Area"]
predicted_probability = prediction_table.iloc[0]["Probability (%)"]

st.success(
    f"Predicted probable area: **{predicted_area}** "
    f"({predicted_probability:.2f}%)"
)


# =========================================================
# SEARCH PRIORITY
# =========================================================

st.header("🚨 Search Priority Areas")

st.dataframe(
    priority[
        [
            "Area",
            "Priority_Score",
            "Priority"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# =========================================================
# ROUTE PREDICTION
# =========================================================

st.header("🛣️ Route Prediction")

current_area = last_record["Area"]

if current_area in route.index:

    next_areas = (
        route.loc[current_area]
        .sort_values(
            ascending=False
        )
        .head(5)
    )

    route_table = pd.DataFrame({
        "Next Area": next_areas.index,
        "Probability (%)":
            next_areas.values * 100
    })

    route_table["Probability (%)"] = (
        route_table["Probability (%)"].round(2)
    )

    st.write(
        f"Current area: **{current_area}**"
    )

    st.dataframe(
        route_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ANOMALIES
# =========================================================

st.header("⚠️ Detected Anomalies")

anomalies = movement[
    movement["Anomaly"] == "Anomaly"
]

st.dataframe(
    anomalies[
        [
            "Person_ID",
            "Latitude",
            "Longitude",
            "Hour",
            "Speed",
            "Distance",
            "Area"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# =========================================================
# MAP
# =========================================================

st.header("🗺️ Interactive Movement Map")

map_center = [
    movement["Latitude"].mean(),
    movement["Longitude"].mean()
]

m = folium.Map(
    location=map_center,
    zoom_start=12
)


# All movement points
for _, row in person_data.iterrows():

    folium.CircleMarker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],
        radius=4,
        popup=(
            f"Area: {row['Area']}<br>"
            f"Hour: {row['Hour']}<br>"
            f"Speed: {row['Speed']}"
        )
    ).add_to(m)


# Anomaly points
for _, row in anomalies.iterrows():

    folium.Marker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],
        popup=(
            f"Anomaly<br>"
            f"Area: {row['Area']}"
        )
    ).add_to(m)


# Last known location
folium.Marker(
    location=[
        last_record["Latitude"],
        last_record["Longitude"]
    ],
    popup="Last Known Location",
    tooltip="Last Known Location"
).add_to(m)


st_folium(
    m,
    width=1200,
    height=600
)


# =========================================================
# EXPLANATION
# =========================================================

st.header("🧠 AI Explanation")

st.write(
    f"""
The system predicts **{predicted_area}** as the most probable
area for the selected movement record.

The prediction is based on movement features including:

- Geographic coordinates
- Movement speed
- Travel distance
- Time of movement
- Movement cluster
- Historical movement patterns

The system also uses anomaly detection and route-transition
patterns to provide additional investigation-support information.
"""
)

st.warning(
    "This system is an academic prototype. "
    "It provides probabilistic analysis and does not prove "
    "a person's actual location."
)

st.caption(
    "CASEFILE AI — Academic Machine Learning Project"
)