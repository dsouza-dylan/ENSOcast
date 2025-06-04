import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import classification_report, confusion_matrix

# --- Config ---
st.set_page_config(page_title="ENSOcast Dashboard", layout="wide")

# --- Load Data and Model ---
@st.cache_data(show_spinner=True)
def load_data_and_model():
    df = pd.read_csv("merged_enso.csv", parse_dates=["Date"])
    model = joblib.load("enso_model_a_baseline.pkl")
    return df, model

@st.cache_data(show_spinner=True)
def load_sst_dataset():
    url = "http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
    ds = xr.open_dataset(url)
    ds['time'] = pd.to_datetime(ds['time'].values)
    return ds

df, model = load_data_and_model()
sst_ds = load_sst_dataset()

# --- Constants ---
label_map = {0: "El Niño", 1: "La Niña", 2: "Neutral"}
label_order = ["El Niño", "La Niña", "Neutral"]

# --- Sidebar Filters ---
st.sidebar.header("Filters")
year_min, year_max = int(df["Date"].dt.year.min()), int(df["Date"].dt.year.max())
year_range = st.sidebar.slider("Year Range", year_min, year_max, (2000, 2020))
phases_selected = st.sidebar.multiselect("ENSO Phases", label_order, default=label_order)

# --- Filter DataFrame ---
df_filtered = df[
    (df["Date"].dt.year >= year_range[0]) &
    (df["Date"].dt.year <= year_range[1]) &
    (df["ENSO_Label"].isin(phases_selected))
].copy()

# --- Features for Prediction ---
feature_cols = [
    "SST_Anomaly", "SOI", "SOI_lag_1", "SOI_lag_2", "SOI_lag_3",
    "SST_Anomaly_lag_1", "SST_Anomaly_lag_2", "SST_Anomaly_lag_3",
    "month_sin", "month_cos"
]

# --- Add Predictions to DataFrames ---
X = df[feature_cols]
y_true = df["ENSO_Label"]
y_pred = model.predict(X)
df["Predicted_Phase"] = [label_map[i] for i in y_pred]
df["True_Phase"] = [label_map[i] for i in y_true]

X_filtered = df_filtered[feature_cols]
y_true_filtered = df_filtered["ENSO_Label"]
y_pred_filtered = model.predict(X_filtered)
df_filtered["Predicted_Phase"] = [label_map[i] for i in y_pred_filtered]
df_filtered["True_Phase"] = [label_map[i] for i in y_true_filtered]

# --- Page Header ---
st.image("ENSOcast_logo_blue.png", width=120)
st.title("🌊 ENSOcast — Your ENSO Forecasting Companion")
st.subheader("Track, Understand, and Forecast ENSO Events with Climate Data")

# --- Tabs ---
tab_sst, tab_trends, tab_model, tab_forecast, tab_download = st.tabs([
    "🌡 SST Snapshot", "📈 Historical Trends", "🤖 Model Insights", "🔮 Forecasting", "📤 Download"
])

# --- Tab 1: SST Snapshot ---
with tab_sst:
    st.header("Sea Surface Temperature (SST) Snapshot")
    selected_year = st.slider("Select Year", year_min, year_max, 2000)
    month_dict = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    selected_month = st.selectbox("Select Month", list(month_dict.keys()), index=7)
    month_num = month_dict[selected_month]

    sst_slice = sst_ds.sel(time=(sst_ds['time.year'] == selected_year) & (sst_ds['time.month'] == month_num))['sst']

    fig, ax = plt.subplots(figsize=(12, 4))
    sst_plot = sst_slice.plot(ax=ax, cmap='coolwarm', cbar_kwargs={"label": "SST (°C)"})

    # Highlight Niño 3.4 region rectangle (approx. 190°E-240°E, 5°S-5°N)
    rect = patches.Rectangle((190, -5), 50, 10, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.text(192, 6, 'Niño 3.4 Region', fontsize=12, color='black')

    st.pyplot(fig)

# --- Tab 2: Historical Trends ---
with tab_trends:
    st.header("Historical SST and ONI Trends")
    metric = st.selectbox("Select Metric", ["SST_Anomaly", "ONI", "Both"], index=2)
    plot_df = df_filtered.copy()

    if metric == "Both":
        cols_to_plot = ["SST_Anomaly", "ONI"]
    else:
        cols_to_plot = [metric]

    fig = px.line(plot_df, x="Date", y=cols_to_plot,
                  labels={"value": "Value", "variable": "Metric"},
                  title="Historical Trends of Selected Metric(s)")

    if "ONI" in cols_to_plot:
        fig.add_hline(y=0.5, line_dash="dot", line_color="red",
                      annotation_text="El Niño Threshold", annotation_position="top left")
        fig.add_hline(y=-0.5, line_dash="dot", line_color="blue",
                      annotation_text="La Niña Threshold", annotation_position="bottom left")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ENSO Phase Predictions Over Time")
    # Line plot of predicted phase with colors
    phase_colors = {"El Niño": "#e74c3c", "La Niña": "#3498db", "Neutral": "#95a5a6"}
    df_filtered['Predicted_Phase_Color'] = df_filtered['Predicted_Phase'].map(phase_colors)
    phase_fig = px.scatter(df_filtered, x="Date", y="Predicted_Phase",
                           color="Predicted_Phase",
                           color_discrete_map=phase_colors,
                           title="Predicted ENSO Phases Timeline",
                           labels={"Predicted_Phase": "Predicted Phase"})
    st.plotly_chart(phase_fig, use_container_width=True)

# --- Tab 3: Model Insights ---
with tab_model:
    st.header("Model Insights")
    report = classification_report(df_filtered["True_Phase"], df_filtered["Predicted_Phase"], output_dict=True)
    accuracy = report.get("accuracy", 0)
    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    st.markdown("### Classification Report")
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df)

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(df_filtered["True_Phase"], df_filtered["Predicted_Phase"], labels=label_order)
    cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in label_order], columns=[f"Pred {l}" for l in label_order])
    st.dataframe(cm_df)

    st.markdown("### Feature Importance")
    importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
    fig2 = px.bar(importance_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                  title="Feature Importance")
    st.plotly_chart(fig2, use_container_width=True)

# --- Tab 4: Forecasting ---
with tab_forecast:
    st.header("Forecast Future ENSO Phases")
    forecast_months = st.slider("Months Ahead to Forecast", 1, 12, 3)
    last_date = df["Date"].max()
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')
    st.write("Future Dates:", future_dates.date.tolist())
    st.info("Forecasting logic coming soon...")

    st.markdown("### Feature Importance (Repeated for Reference)")
    importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
    fig3 = px.bar(importance_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig3, use_container_width=True)

# --- Tab 5: Download ---
with tab_download:
    st.header("Download Data")
    st.download_button("Download Filtered Predictions CSV", data=df_filtered.to_csv(index=False), file_name="enso_predictions_filtered.csv")
    st.download_button("Download Full Dataset CSV", data=df.to_csv(index=False), file_name="enso_predictions_full.csv")

