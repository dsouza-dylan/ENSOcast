# ENSOcast: Refined and Narrative-Driven Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------- Page Config ---------------------
st.set_page_config(page_title="ENSOcast", layout="wide")

# --------------------- Load Data -----------------------
@st.cache_data
def load_enso_data():
    df = pd.read_csv("merged_enso.csv", parse_dates=["Date"])
    return df

@st.cache_data
def load_sst_dataset():
    url = "http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
    ds = xr.open_dataset(url)
    ds["time"] = pd.to_datetime(ds["time"].values)
    return ds

# --------------------- Data Prep -----------------------
df = load_enso_data()
sst_ds = load_sst_dataset()

feature_cols = [
    "SST_Anomaly", "SOI", "SOI_lag_1", "SOI_lag_2", "SOI_lag_3",
    "SST_Anomaly_lag_1", "SST_Anomaly_lag_2", "SST_Anomaly_lag_3",
    "month_sin", "month_cos"
]

label_map = {0: "La Niña", 1: "Neutral", 2: "El Niño"}

# --------------------- Sidebar -------------------------
st.sidebar.title("🌊 ENSOcast")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🧭 Overview",
    "📈 ENSO Historical Signals",
    "🌡 Global SST Snapshot",
    "🛠 Train Your Own Forecast Model"
])
st.sidebar.markdown("---")
st.sidebar.markdown("Made by Dylan Dsouza")

# --------------------- Overview Page -------------------
if page == "🧭 Overview":
    st.title("🧭 Welcome to ENSOcast")
    st.markdown("""
    **ENSOcast** is a forecasting dashboard built to explore and model the **El Niño-Southern Oscillation (ENSO)**. 

    🔍 Use historical indicators like **ONI, SOI, and SST anomalies** to understand ENSO trends.
    
    🛠 Train a custom model to predict ENSO phases and evaluate its performance interactively.
    
    🌐 Visualize real global SST maps from NOAA's high-resolution dataset.

    All data sources: NOAA PSL, ONI index, and more.
    """)

# --------------------- Historical Signals --------------
elif page == "📈 ENSO Historical Signals":
    st.header("📈 Historical ENSO Trends")

    years = st.slider("Select Year Range", 1982, 2025, (2000, 2020))
    selected_phases = st.multiselect("Select ENSO Phases", ["La Niña", "Neutral", "El Niño"], default=["La Niña", "Neutral", "El Niño"])

    df_filtered = df[(df["Date"].dt.year.between(years[0], years[1])) & (df["ENSO_Phase"].isin(selected_phases))]

    st.subheader("SST and ONI Timeline")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["SST_Anomaly"], name="SST Anomaly (Niño 3.4)", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_filtered["Date"], y=df_filtered["ONI"], name="ONI", line=dict(color='dodgerblue')))
    fig.update_layout(title="Niño 3.4 SST Anomaly vs ONI", xaxis_title="Date", yaxis_title="Anomaly", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Southern Oscillation Index (SOI)")
    fig_soi = px.line(df_filtered, x="Date", y="SOI", title="SOI Timeline", template="plotly_dark")
    st.plotly_chart(fig_soi, use_container_width=True)

# --------------------- SST Snapshot --------------------
elif page == "🌡 Global SST Snapshot":
    st.header("🌡 Global Sea Surface Temperature (SST) Snapshot")
    selected_year = st.slider("Select Year", min_value=1982, max_value=2024, value=2010)
    selected_month = st.selectbox("Select Month", list(range(1, 13)), index=7)

    try:
        sst_slice = sst_ds.sel(time=(sst_ds['time.year'] == selected_year) & (sst_ds['time.month'] == selected_month))['sst']
        fig, ax = plt.subplots(figsize=(12, 6))
        sst_slice.plot(ax=ax, cmap='coolwarm', cbar_kwargs={"label": "°C"})
        ax.add_patch(patches.Rectangle((190, -5), 50, 10, edgecolor='black', facecolor='none', linewidth=1))
        ax.text(189, 8, 'Niño 3.4 Region', color='black')
        ax.set_xlabel("Longitude [°E]")
        ax.set_ylabel("Latitude [°N]")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to fetch SST data for {selected_month}/{selected_year}. Error: {e}")

# --------------------- Train Forecast Model ------------
elif page == "🛠 Train Your Own Forecast Model":
    st.header("🛠 Train Your Own ENSO Forecast Model")

    years = st.slider("Select Year Range for Training", 1982, 2025, (2000, 2020))
    selected_phases = st.multiselect("Filter by ENSO Phase", ["La Niña", "Neutral", "El Niño"], default=["La Niña", "Neutral", "El Niño"])

    df_custom = df[(df["Date"].dt.year >= years[0]) & (df["Date"].dt.year <= years[1]) & (df["ENSO_Phase"].isin(selected_phases))]

    X_custom = df_custom[feature_cols]
    y_custom = df_custom["ENSO_Label"]
    X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.3, shuffle=False)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Custom Model Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=label_map.values(), output_dict=True)).transpose()
    st.dataframe(report_df.round(2))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm, index=label_map.values(), columns=label_map.values()))

    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    fig_feat = px.bar(importance_df, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig_feat, use_container_width=True)

    df_custom["Predicted_Phase"] = [label_map[i] for i in y_pred]
    df_custom["True_Phase"] = [label_map[i] for i in y_test]

    st.subheader("📥 Download Custom Predictions")
    st.download_button("Download CSV", data=df_custom.to_csv(index=False), file_name="custom_enso_predictions.csv", mime="text/csv")
