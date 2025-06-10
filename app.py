import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="ENSOcast", layout="wide")

# Constants
FEATURE_COLS = [
    "SST_Anomaly", "SOI", "SOI_lag_1", "SOI_lag_2", "SOI_lag_3",
    "SST_Anomaly_lag_1", "SST_Anomaly_lag_2", "SST_Anomaly_lag_3",
    "month_sin", "month_cos"
]
LABEL_MAP = {0: "La Niña", 1: "Neutral", 2: "El Niño"}
MONTH_DICT = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

@st.cache_data
def load_data():
    """Load and preprocess all data"""
    df = pd.read_csv("merged_enso.csv", parse_dates=["Date"])
    model = joblib.load("enso_model_a_baseline.pkl")

    # Add predictions to dataframe
    X = df[FEATURE_COLS]
    y_pred = model.predict(X)
    df["Predicted_Phase"] = [LABEL_MAP[i] for i in y_pred]
    df["True_Phase"] = [LABEL_MAP[i] for i in df["ENSO_Label"]]

    return df, model

@st.cache_data
def load_sst_dataset():
    """Load SST dataset"""
    url = "http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
    ds = xr.open_dataset(url)
    ds["time"] = pd.to_datetime(ds["time"].values)
    return ds

def filter_data(df, years, phases):
    """Common data filtering function"""
    return df[
        (df["Date"].dt.year >= years[0]) &
        (df["Date"].dt.year <= years[1]) &
        (df["True_Phase"].isin(phases))
    ]

def create_sst_plot(sst_ds, year, month):
    """Create SST visualization"""
    try:
        sst_slice = sst_ds.sel(time=(sst_ds['time.year'] == year) & (sst_ds['time.month'] == month))['sst']
        fig, ax = plt.subplots(figsize=(12, 6))
        sst_slice.plot(ax=ax, cmap='coolwarm', cbar_kwargs={"label": "°C"})
        ax.add_patch(patches.Rectangle((190, -5), 50, 10, edgecolor='black', facecolor='none', linewidth=1))
        ax.text(189, 8, 'Niño 3.4 Region', color='black')
        ax.set_xlabel("Longitude [°E]")
        ax.set_ylabel("Latitude [°N]")
        return fig
    except Exception as e:
        st.error(f"Failed to fetch SST data. Error: {e}")
        return None

def create_timeline_plots(df_filtered):
    """Create all timeline plots"""
    # SST Timeline
    st.markdown("### Sea Surface Temperature (SST) Timeline")
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df_filtered["Date"], y=df_filtered["SST_Climatology"],
        name="Climatological SST (°C)", line=dict(color='deepskyblue', dash='dot')
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df_filtered["Date"], y=df_filtered["SST"],
        name="Observed SST (°C)", line=dict(color='orange')
    ), secondary_y=False)

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Sea Surface Temperature (°C)",
        legend=dict(x=0.01, y=1.1), template="plotly_dark", margin=dict(t=30, b=30)
    )
    fig.update_yaxes(title_text="Sea Surface Temperature (°C)", range=[24, 30], secondary_y=False)
    fig.update_yaxes(range=[24, 30], secondary_y=True, showticklabels=False)

    # Add climatology lines
    clim_min, clim_max = df_filtered["SST_Climatology"].min(), df_filtered["SST_Climatology"].max()
    fig.add_hline(y=clim_min, line_dash="dot", line_color="gray", annotation_text="Min. Climatological SST")
    fig.add_hline(y=clim_max, line_dash="dot", line_color="gray", annotation_text="Max. Climatological SST")

    st.plotly_chart(fig, use_container_width=True)

    # ONI Timeline
    st.markdown("### Oceanic Niño Index (ONI) Timeline")
    fig_oni = px.line(df_filtered, x="Date", y="ONI")
    fig_oni.update_yaxes(title_text="Oceanic Niño Index")
    fig_oni.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="El Niño Threshold")
    fig_oni.add_hline(y=-0.5, line_dash="dot", line_color="blue", annotation_text="La Niña Threshold")
    st.plotly_chart(fig_oni, use_container_width=True)

    # SOI Timeline
    st.markdown("### Southern Oscillation Index (SOI) Timeline")
    fig_soi = px.line(df_filtered, x="Date", y="SOI")
    fig_soi.update_layout(yaxis_title="Southern Oscillation Index", template="plotly_dark", margin=dict(t=30, b=30))
    fig_soi.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_soi.add_hrect(y0=0, y1=3.2, line_width=0, fillcolor="blue", opacity=0.1)
    fig_soi.add_hrect(y0=-3.2, y1=0, line_width=0, fillcolor="red", opacity=0.1)
    st.plotly_chart(fig_soi, use_container_width=True)

# Load data
df, model = load_data()
sst_ds = load_sst_dataset()

# Main UI
st.title("🌊 ENSOcast: El Niño–Southern Oscillation Forecasts")

# Sidebar
st.sidebar.title("🌊 ENSOcast")
st.sidebar.markdown("---")
page = st.sidebar.radio("📂 Tab Navigation", [
    "🌡 Global SST Snapshot",
    "📈 Historical Trends",
    "🛠 Interactive Prediction Tool"
], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("Made by Dylan Dsouza")

# Page routing
if page == "🌡 Global SST Snapshot":
    st.header("🌡 Global Sea Surface Temperature (SST) Snapshot")

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.slider("Select Year", min_value=1982, max_value=2024, value=2010)
    with col2:
        selected_month = st.selectbox("Select Month", list(MONTH_DICT.keys()), index=7)

    fig = create_sst_plot(sst_ds, selected_year, MONTH_DICT[selected_month])
    if fig:
        st.pyplot(fig)

elif page == "📈 Historical Trends":
    st.header("📈 Historical Trends")

    col1, col2 = st.columns(2)
    with col1:
        years = st.slider("Select Year Range", 1982, 2025, (2000, 2020))
    with col2:
        selected_phases = st.multiselect("Select ENSO Phases",
                                       ["La Niña", "Neutral", "El Niño"],
                                       default=["La Niña", "Neutral", "El Niño"])

    df_filtered = filter_data(df, years, selected_phases)
    create_timeline_plots(df_filtered)

elif page == "🛠 Interactive Prediction Tool":
    st.header("🛠 Interactive Prediction Tool")

    col1, col2 = st.columns(2)
    with col1:
        years = st.slider("Select Year Range", 1982, 2025, (2000, 2020))
    with col2:
        selected_phases = st.multiselect("Select ENSO Phases",
                                       ["La Niña", "Neutral", "El Niño"],
                                       default=["La Niña", "Neutral", "El Niño"])

    filtered_df = filter_data(df, years, selected_phases)

    # Train custom model
    X_custom = filtered_df[FEATURE_COLS]
    y_custom = filtered_df["True_Phase"]
    X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.3, shuffle=False)

    custom_model = RandomForestClassifier(random_state=42)
    custom_model.fit(X_train, y_train)
    y_pred_custom = custom_model.predict(X_test)

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Custom Model Accuracy", f"{accuracy_score(y_test, y_pred_custom) * 100:.2f}%")

    # Classification report
    st.markdown("### Classification Report")
    report = classification_report(y_test, y_pred_custom, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    # Confusion matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_custom, labels=["La Niña", "Neutral", "El Niño"])
    st.dataframe(pd.DataFrame(cm,
                             index=["True La Niña", "True Neutral", "True El Niño"],
                             columns=["Pred La Niña", "Pred Neutral", "Pred El Niño"]))

    # Feature importance
    st.markdown("### Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": custom_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    # Download results
    filtered_df["Predicted_Phase"] = custom_model.predict(X_custom)
    st.markdown("### Download Results")
    st.download_button("📥 Download CSV",
                      filtered_df.to_csv(index=False),
                      "custom_enso_predictions.csv",
                      mime="text/csv")
