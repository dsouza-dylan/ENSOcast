import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load data and model
df = pd.read_csv("merged_enso.csv", parse_dates=["Date"])
model = joblib.load("best_rf_model.pkl")

st.set_page_config(page_title="ENSO ML Dashboard", layout="wide")

st.title("🌊 ENSO Forecasting with Machine Learning")
st.markdown("Using Sea Surface Temperature and ONI data to classify El Niño, La Niña, and Neutral events.")

# Sidebar controls
st.sidebar.header("Filters")
year_range = st.sidebar.slider("Select Year Range", 1981, 2025, (2000, 2020))
phases = st.sidebar.multiselect("Select ENSO Phases", ["El Niño", "La Niña", "Neutral"], default=["El Niño", "La Niña", "Neutral"])

# Filter data
df = df[(df["Date"].dt.year >= year_range[0]) & (df["Date"].dt.year <= year_range[1])]
df = df[df["ENSO_Phase"].isin(phases)]

# ---- Main Section ----
st.subheader("📈 SST and ONI Time Series")
fig = px.line(df, x="Date", y=["sst", "ONI"], labels={"value": "Temperature / Index", "variable": "Metric"}, title="SST and ONI Over Time")
st.plotly_chart(fig, use_container_width=True)

# ---- Model Prediction & Evaluation ----
st.subheader("🧠 Model Prediction & Evaluation")

features = ['sst_anomaly', 'oni_lag_1', 'oni_lag_2', 'oni_lag_3', 'month_sin', 'month_cos']
X = df[features]
y_true = df["ENSO_Label"]
y_pred = model.predict(X)

label_map = {0: "El Niño", 1: "La Niña", 2: "Neutral"}
y_true_labels = [label_map[i] for i in y_true]
y_pred_labels = [label_map[i] for i in y_pred]

report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)

st.dataframe(report_df)

# ---- Confusion Matrix ----
st.subheader("🔁 Confusion Matrix")
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=["El Niño", "La Niña", "Neutral"])
cm_df = pd.DataFrame(cm, index=["True El Niño", "True La Niña", "True Neutral"],
                        columns=["Pred El Niño", "Pred La Niña", "Pred Neutral"])
st.dataframe(cm_df)

# ---- Feature Importances ----
st.subheader("📌 Feature Importances")
importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values("Importance", ascending=False)

fig_imp = px.bar(feat_df, x="Importance", y="Feature", orientation="h", title="Feature Importances")
st.plotly_chart(fig_imp, use_container_width=True)

# ---- Footer ----
st.markdown("---")
st.markdown("✅ Built with NOAA data | Model: Random Forest | Author: You :)")
