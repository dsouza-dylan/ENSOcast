import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# --- Config ---
st.set_page_config(page_title="ENSOcast Dashboard", layout="wide")

# --- Load Model and Data ---
@st.cache_data
def load_model_and_data():
    df = pd.read_csv("merged_enso.csv", parse_dates=["Date"])
    model = joblib.load("enso_model_a_baseline.pkl")
    return df, model

df, model = load_model_and_data()

# --- Preprocess for Predictions ---
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
st.subheader("Track, Understand, and Forecast ENSO Events with Climate Data")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔎 Model Insights", "📤 Download"])

# --- Tab 1: Trends ---
with tab1:
    st.markdown("### SST Anomalies Over Time")
    fig = px.line(df, x="Date", y="SST_Anomaly", title="SST Anomaly Timeline", labels={"SST_Anomaly": "SST Anomaly (°C)"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ENSO Phase Predictions")
    fig2 = px.line(df, x="Date", y="Predicted_Phase", title="Predicted ENSO Phase Timeline", color_discrete_sequence=["#e74c3c", "#3498db", "#95a5a6"])
    st.plotly_chart(fig2, use_container_width=True)

# --- Tab 2: Model Insights ---
with tab2:
    st.markdown("### Classification Report")
    report = classification_report(df["True_Phase"], df["Predicted_Phase"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(df["True_Phase"], df["Predicted_Phase"], labels=["El Niño", "La Niña", "Neutral"])
    st.dataframe(pd.DataFrame(cm, index=["True El Niño", "True La Niña", "True Neutral"], columns=["Pred El Niño", "Pred La Niña", "Pred Neutral"]))

    st.markdown("### Feature Importance")
    importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
    fig3 = px.bar(importance_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig3, use_container_width=True)

# --- Tab 3: Download ---
with tab3:
    st.markdown("### Download Full Dataset with Predictions")
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="enso_predictions.csv", mime="text/csv")
