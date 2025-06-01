import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import classification_report, confusion_matrix

# --- Load Data ---
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv("merged_enso.csv", parse_dates=["Date"])
    model = joblib.load("best_rf_model.pkl")
    return df, model

@st.cache_data(show_spinner=True)
def load_sst_dataset():
    url = "http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
    ds = xr.open_dataset(url)
    ds['time'] = pd.to_datetime(ds['time'].values)
    return ds

# --- Initial Setup ---
df, model = load_data()
sst_ds = load_sst_dataset()
label_order = ["El Niño", "La Niña", "Neutral"]
label_map = {0: "El Niño", 1: "La Niña", 2: "Neutral"}

st.set_page_config(page_title="ENSOcast", layout="wide")

# --- Header ---
st.image("ENSOcast_logo_blue.png", width=120)
st.title("ENSOcast — Your ENSO Forecasting Companion")

# --- Sidebar ---
st.sidebar.header("Filters")
year_range = st.sidebar.slider("Year Range", 1981, 2025, (2000, 2020))
phases = st.sidebar.multiselect("ENSO Phases", label_order, default=label_order)

df = df[(df["Date"].dt.year >= year_range[0]) & (df["Date"].dt.year <= year_range[1])]
df = df[df["ENSO_Phase"].isin(phases)]

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🌡 SST Snapshot", "📈 Historical Trends", "🤖 Model Insights", "🔮 Forecasting"])

# --- Tab 1: SST Snapshot ---
with tab1:
    st.subheader("Global SST Snapshot")
    selected_year = st.slider("Year", 1981, 2025, 2000)
    month_dict = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                  "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
    selected_month = st.selectbox("Month", list(month_dict.keys()), index=7)
    month_num = month_dict[selected_month]

    sst_slice = sst_ds.sel(time=(sst_ds['time.year'] == selected_year) & (sst_ds['time.month'] == month_num))['sst']
    fig, ax = plt.subplots(figsize=(12, 4))
    sst_slice.plot(ax=ax, cmap='coolwarm', cbar_kwargs={"label": "°C"})
    ax.add_patch(patches.Rectangle((190, -5), 50, 10, edgecolor='black', facecolor='none'))
    ax.text(192, 6, 'Niño 3.4 Region', color='black')
    st.pyplot(fig)

# --- Tab 2: Historical Trends ---
with tab2:
    st.subheader("Historical SST and ONI Trends")
    metric = st.selectbox("Metric", ["sst", "ONI", "Both"])
    cols = ["sst", "ONI"] if metric == "Both" else [metric]
    fig = px.line(df, x="Date", y=cols, labels={"value": "Index", "variable": "Metric"})
    if "ONI" in cols:
        fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="El Niño Threshold")
        fig.add_hline(y=-0.5, line_dash="dot", line_color="blue", annotation_text="La Niña Threshold")
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Model Insights ---
with tab3:
    st.subheader("Model Insights")
    X = df[['sst_anomaly', 'oni_lag_1', 'oni_lag_2', 'oni_lag_3', 'month_sin', 'month_cos']]
    y_true = df["ENSO_Label"]
    y_pred = model.predict(X)
    y_true_lbls = [label_map[i] for i in y_true]
    y_pred_lbls = [label_map[i] for i in y_pred]

    report = classification_report(y_true_lbls, y_pred_lbls, output_dict=True)
    st.metric("Accuracy", f"{report['accuracy'] * 100:.2f}%")
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    cm = confusion_matrix(y_true_lbls, y_pred_lbls, labels=label_order)
    st.dataframe(pd.DataFrame(cm, index=label_order, columns=label_order))

    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    fig = px.bar(importance_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    df["Predicted_Phase"] = y_pred_lbls
    st.download_button("Download Predictions", df.to_csv(index=False), "enso_predictions.csv")

# --- Tab 4: Forecasting ---
with tab4:
    st.subheader("Forecast Future ENSO Phases")
    forecast_months = st.slider("Months Ahead", 1, 12, 3)
    last_date = df["Date"].max()
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')
    st.write("Future Dates:", future_dates.date.tolist())

    # Additional logic to generate realistic future predictions would go here...
