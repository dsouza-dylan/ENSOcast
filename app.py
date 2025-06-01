import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import xarray as xr
import matplotlib.pyplot as plt

df = pd.read_csv("merged_enso.csv", parse_dates=["Date"])
model = joblib.load("best_rf_model.pkl")

label_order = ["El Niño", "La Niña", "Neutral"]
label_map = {0: "El Niño", 1: "La Niña", 2: "Neutral"}

st.set_page_config(page_title="ENSOcast", layout="wide")

# === TITLE & INTRO ===
st.image("ENSOcast_logo_blue.png", width=120)
st.subheader("🌎 Understanding ENSO (El Niño—Southern Oscillation)")
st.title("ENSOcast — Your ENSO Forecasting Companion")
st.markdown("""  
ENSO stands for El—Niño Southern Oscillation, a natural climate pattern characterized by fluctuations in sea surface temperatures and atmospheric pressure in the tropical Pacific Ocean.  
These fluctuations strongly influence global weather, affecting rainfall, droughts, and marine ecosystems worldwide.

This app uses machine learning to classify ENSO phases — *El Niño*, *La Niña*, and *Neutral* — based on Sea Surface Temperature (SST) and Oceanic Niño Index (ONI) data.
Explore historical trends, global SST snapshots, and model predictions to understand ENSO dynamics and their global impacts.
""")

# === SIDEBAR FILTERS ===
st.sidebar.header("Filters")
year_range = st.sidebar.slider("Select Year Range", 1981, 2025, (2000, 2020), help="Filter data by year range")
phases = st.sidebar.multiselect("Select ENSO Phases", label_order, default=label_order, help="Choose which ENSO phases to display")

df = df[(df["Date"].dt.year >= year_range[0]) & (df["Date"].dt.year <= year_range[1])]
df = df[df["ENSO_Phase"].isin(phases)]

# === GLOBAL SST SNAPSHOT ===
@st.cache_data(show_spinner=True)
def load_sst_dataset():
    url = "http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
    ds = xr.open_dataset(url)
    ds['time'] = pd.to_datetime(ds['time'].values)
    august_sst = ds.sel(time=ds['time.month'] == 8)
    return august_sst

st.subheader("🌡️ Global Sea Surface Temperature Snapshot (August)")
st.markdown("""
This map shows sea surface temperatures (SST) globally for August of the selected year.  
Warm areas (red) often indicate El Niño conditions, while cooler areas (blue) suggest La Niña.  
Monitoring SST patterns helps predict ENSO phases and their potential impacts worldwide.
""")

sst_ds = load_sst_dataset()
available_years = pd.DatetimeIndex(sst_ds.time.values).year
selected_year = st.slider("Select Year for SST Snapshot", int(available_years.min()), int(available_years.max()), 2000)

sst_slice = sst_ds.sel(time=str(selected_year))['sst']

fig, ax = plt.subplots(figsize=(12, 4))
sst_slice.plot(ax=ax, cmap='coolwarm', cbar_kwargs={"label": "Temperature (°C)"})
ax.set_title(f"Sea Surface Temperature - August {selected_year}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
st.pyplot(fig)

# === TIME SERIES VISUALIZATION ===
st.subheader("📈 Historical SST and ONI Trends")
st.markdown("""
Explore historical trends in Sea Surface Temperature (SST) and Oceanic Niño Index (ONI), which are key indicators used to identify ENSO phases.  
Use the filters above to customize the timeframe and ENSO phases displayed.
""")

metric_option = st.selectbox("Select Metric to Plot", ["sst", "ONI", "Both"], help="Choose which metrics to visualize over time")
plot_cols = ["sst", "ONI"] if metric_option == "Both" else [metric_option]
fig = px.line(df, x="Date", y=plot_cols, labels={"value": "Temperature / Index", "variable": "Metric"},
              title="SST and ONI Over Time")
st.plotly_chart(fig, use_container_width=True)

# === MODEL PREDICTIONS AND EVALUATION ===
st.subheader("🤖 ENSO Phase Predictions & Model Performance")
st.markdown("""
The Random Forest model predicts ENSO phases based on SST anomalies and recent ONI values, capturing seasonal patterns through sine and cosine transformations of months.  
Below are the prediction accuracy and detailed classification metrics for each ENSO phase.
""")

features = ['sst_anomaly', 'oni_lag_1', 'oni_lag_2', 'oni_lag_3', 'month_sin', 'month_cos']
X = df[features]
y_true = df["ENSO_Label"]
y_pred = model.predict(X)

y_true_labels = [label_map[i] for i in y_true]
y_pred_labels = [label_map[i] for i in y_pred]

report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)
st.metric("Overall Model Accuracy", f"{report['accuracy'] * 100:.2f}%")

st.markdown("### 📊 Classification Report Summary")
st.dataframe(report_df.loc[label_order + ["accuracy"]])

st.subheader("🔁 Confusion Matrix")
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_order)
cm_df = pd.DataFrame(cm, index=["True " + lbl for lbl in label_order],
                        columns=["Pred " + lbl for lbl in label_order])
st.dataframe(cm_df)

# === FEATURE IMPORTANCE ===
st.subheader("🧩 Feature Importance")
st.markdown("""
Understanding which features most influence the model's predictions helps interpret the results and improve model trust.  
Here, SST anomalies and recent ONI lag values play the largest roles.
""")
importances = model.feature_importances_
feat_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)
fig_imp = px.bar(feat_df, x="Importance", y="Feature", orientation="h", title="Feature Importances")
st.plotly_chart(fig_imp, use_container_width=True)

# === DOWNLOAD PREDICTIONS ===
st.subheader("📥 Download Predicted ENSO Phases")
df["Predicted_Phase"] = y_pred_labels
st.download_button("Download Predictions as CSV", df.to_csv(index=False), file_name="enso_predictions.csv", mime='text/csv')

# === MODEL DETAILS & CLOSING ===
st.markdown("---")
st.markdown("### 🌳 Model Details")
if hasattr(model, 'n_estimators'):
    st.write(f"Number of Trees: {model.n_estimators}")
if hasattr(model, 'max_depth'):
    st.write(f"Max Depth: {model.max_depth}")

st.markdown("---")
st.markdown("✅ Built with NOAA data | Model: Random Forest | Author: You :)")
