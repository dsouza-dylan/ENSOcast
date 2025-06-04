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
st.set_page_config(page_title="ENSOcast", layout="wide")

# --- Load Data ---
@st.cache_data
def load_model_and_data():
    df = pd.read_csv("merged_enso.csv", parse_dates=["Date"])
    model = joblib.load("enso_model_a_baseline.pkl")
    return df, model

@st.cache_data
def load_sst_dataset():
    url = "http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
    ds = xr.open_dataset(url)
    ds["time"] = pd.to_datetime(ds["time"].values)
    return ds

df, model = load_model_and_data()
sst_ds = load_sst_dataset()

# --- Features and Labels ---
feature_cols = [
    "SST_Anomaly", "SOI", "SOI_lag_1", "SOI_lag_2", "SOI_lag_3",
    "SST_Anomaly_lag_1", "SST_Anomaly_lag_2", "SST_Anomaly_lag_3",
    "month_sin", "month_cos"
]
X = df[feature_cols]
y_true = df["ENSO_Label"]
y_pred = model.predict(X)

label_map = {0: "El Niño", 1: "La Niña", 2: "Neutral"}
df["Predicted_Phase"] = [label_map[i] for i in y_pred]
df["True_Phase"] = [label_map[i] for i in y_true]

# --- Header ---
st.title("🌊 ENSOcast")
st.subheader("Track, Understand, and Forecast ENSO Events")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🌡 SST Snapshot", "📈 Trends", "🔎 Model Insights", "📤 Download"])

# --- Tab 1: SST Snapshot ---
with tab1:
    st.markdown("### Global SST Snapshot")
    selected_year = st.slider("Select Year", min_value=1982, max_value=2024, value=2010)
    month_dict = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    selected_month = st.selectbox("Select Month", list(month_dict.keys()), index=7)
    month_num = month_dict[selected_month]


    try:
        sst_slice = sst_ds.sel(time=(sst_ds['time.year'] == selected_year) & (sst_ds['time.month'] == month_num))['sst']
        fig, ax = plt.subplots(figsize=(12, 4))
        sst_slice.plot(ax=ax, cmap='coolwarm', cbar_kwargs={"label": "°C"})
        ax.add_patch(patches.Rectangle((190, -5), 50, 10, edgecolor='black', facecolor='none', linewidth=2))
        ax.text(192, 6, 'Niño 3.4 Region', color='black')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to fetch SST data for {selected_month} {selected_year}. Error: {e}")

# --- Tab 2: Trends ---
with tab2:
    st.markdown("### SST Anomaly Timeline")
    fig = px.line(df, x="Date", y="SST_Anomaly", labels={"SST_Anomaly": "SST Anomaly (°C)"})
    st.plotly_chart(fig, use_container_width=True)

    # st.subheader("### ONI Timeline")
    # fig_oni = px.line(df, x="Date", y="ONI", title="ONI (Oceanic Niño Index) Over Time", labels={"oni": "ONI Value"})
    # fig_oni.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="El Niño Threshold", annotation_position="bottom right")
    # fig_oni.add_hline(y=-0.5, line_dash="dot", line_color="blue", annotation_text="La Niña Threshold", annotation_position="top right")
    # st.plotly_chart(fig_oni, use_container_width=True)
    st.subheader("ONI Timeline")

    fig_oni = px.line(df, x="Date", y="oni", title="ONI (Oceanic Niño Index) Over Time", labels={"oni": "ONI Value"})

    # Add threshold lines
    fig_oni.add_hline(y=0.5, line_dash="dot", line_color="red")
    fig_oni.add_hline(y=-0.5, line_dash="dot", line_color="blue")

    # Add custom annotations OUTSIDE the plot
    fig_oni.add_annotation(
        x=df["Date"].max(), y=0.5,
        xref="x", yref="y",
        text="El Niño Threshold (+0.5)",
        showarrow=False,
        font=dict(color="red"),
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.8)"
    )
    fig_oni.add_annotation(
        x=df["Date"].max(), y=-0.5,
        xref="x", yref="y",
        text="La Niña Threshold (−0.5)",
        showarrow=False,
        font=dict(color="blue"),
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)"
    )

    st.plotly_chart(fig_oni, use_container_width=True)


    st.markdown("### Predicted ENSO Phase")
    fig2 = px.line(df, x="Date", y="Predicted_Phase", color_discrete_sequence=["#e74c3c", "#3498db", "#95a5a6"])
    st.plotly_chart(fig2, use_container_width=True)

# --- Tab 3: Model Insights ---
with tab3:
    st.markdown("### Classification Report")
    report = classification_report(df["True_Phase"], df["Predicted_Phase"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(df["True_Phase"], df["Predicted_Phase"], labels=["El Niño", "La Niña", "Neutral"])
    st.dataframe(pd.DataFrame(cm, index=["True El Niño", "True La Niña", "True Neutral"], columns=["Pred El Niño", "Pred La Niña", "Pred Neutral"]))

    st.markdown("### Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    fig3 = px.bar(importance_df, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig3, use_container_width=True)

# --- Tab 4: Download ---
with tab4:
    st.markdown("### Download Predictions CSV")
    st.download_button("📥 Download ENSO Predictions", data=df.to_csv(index=False), file_name="enso_predictions.csv", mime="text/csv")
