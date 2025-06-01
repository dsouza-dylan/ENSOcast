import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load data and model
df = pd.read_csv("merged_enso.csv", parse_dates=["Date"])
model = joblib.load("best_rf_model.pkl")

# ENSO phase label map setup
label_order = ["El Niño", "La Niña", "Neutral"]
label_map = {0: "El Niño", 1: "La Niña", 2: "Neutral"}

# Streamlit page setup
st.set_page_config(page_title="ENSO ML Dashboard", layout="wide")
st.title("🌊 ENSO Forecasting with Machine Learning")
st.markdown("Using Sea Surface Temperature and ONI data to classify El Niño, La Niña, and Neutral events.")

# Sidebar filters
st.sidebar.header("Filters")
year_range = st.sidebar.slider("Select Year Range", 1981, 2025, (2000, 2020))
phases = st.sidebar.multiselect("Select ENSO Phases", label_order, default=label_order)

# Filter data
df = df[(df["Date"].dt.year >= year_range[0]) & (df["Date"].dt.year <= year_range[1])]
df = df[df["ENSO_Phase"].isin(phases)]

# Time series plot with user selection
st.subheader("📈 SST and ONI Time Series")
metric_option = st.selectbox("Select Metric to Plot", ["sst", "ONI", "Both"])
plot_cols = ["sst", "ONI"] if metric_option == "Both" else [metric_option]
fig = px.line(df, x="Date", y=plot_cols, labels={"value": "Temperature / Index", "variable": "Metric"},
              title="SST and ONI Over Time")
st.plotly_chart(fig, use_container_width=True)

# Model prediction
st.subheader("🧬 Model Prediction & Evaluation")
features = ['sst_anomaly', 'oni_lag_1', 'oni_lag_2', 'oni_lag_3', 'month_sin', 'month_cos']
X = df[features]
y_true = df["ENSO_Label"]
y_pred = model.predict(X)

# Convert to readable labels
y_true_labels = [label_map[i] for i in y_true]
y_pred_labels = [label_map[i] for i in y_pred]

# Classification report
report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)
st.metric("Overall Model Accuracy", f"{report['accuracy'] * 100:.2f}%")

st.markdown("### 📊 Classification Report Summary")
st.dataframe(report_df.loc[label_order + ["accuracy"]])

# Confusion matrix
st.subheader("🔁 Confusion Matrix")
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_order)
cm_df = pd.DataFrame(cm, index=["True " + lbl for lbl in label_order],
                        columns=["Pred " + lbl for lbl in label_order])
st.dataframe(cm_df)

# Feature importances
st.subheader("📌 Feature Importances")
importances = model.feature_importances_
feat_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)
fig_imp = px.bar(feat_df, x="Importance", y="Feature", orientation="h", title="Feature Importances")
st.plotly_chart(fig_imp, use_container_width=True)

# GLobe
import xarray as xr
import matplotlib.pyplot as plt

@st.cache_data(show_spinner=True)
def load_sst_dataset():
    url = "http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
    ds = xr.open_dataset(url)
    return ds

st.subheader("🌡️ SST Monthly Snapshot Visualization")

sst_ds = load_sst_dataset()

# Let user select a time index via slider
time_idx = st.slider("Select Time Index (Month)", 0, len(sst_ds.time)-1, 0)

# Extract date string for the selected time
time_str = str(sst_ds.time[time_idx].values)[:10]

# Select SST slice for the chosen time index
sst_slice = sst_ds['sst'].isel(time=time_idx)

# Plot using matplotlib figure and show in Streamlit
fig, ax = plt.subplots(figsize=(10,4))
sst_slice.plot(ax=ax, cmap='coolwarm')
ax.set_title(f"Sea Surface Temperature on {time_str}")
st.pyplot(fig)

# Download predictions
st.subheader("📅 Download Predictions")
df["Predicted_Phase"] = y_pred_labels
st.download_button("📥 Download Predictions as CSV", df.to_csv(index=False), file_name="enso_predictions.csv")

# Model info (optional)
st.markdown("---")
st.markdown("### 🌳 Model Details")
if hasattr(model, 'n_estimators'):
    st.write(f"Number of Trees: {model.n_estimators}")
if hasattr(model, 'max_depth'):
    st.write(f"Max Depth: {model.max_depth}")


st.markdown("---")
st.markdown("✅ Built with NOAA data | Model: Random Forest | Author: You :)")
