import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="ENSOcast", layout="wide", initial_sidebar_state="expanded")

# Constants
FEATURE_COLS = [
    "SST_Anomaly", "SOI", "SOI_lag_1", "SOI_lag_2", "SOI_lag_3",
    "SST_Anomaly_lag_1", "SST_Anomaly_lag_2", "SST_Anomaly_lag_3",
    "month_sin", "month_cos"
]
LABEL_MAP = {0: "La Niña", 1: "Neutral", 2: "El Niño"}
PHASE_COLORS = {"La Niña": "#3498db", "Neutral": "#95a5a6", "El Niño": "#e74c3c"}

@st.cache_data
def load_data():
    """Load and preprocess data"""
    df = pd.read_csv("merged_enso.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    # Load model and make predictions
    model = joblib.load("enso_model_a_baseline.pkl")
    X = df[FEATURE_COLS]
    y_pred = model.predict(X)

    # Add predictions to dataframe
    df["Predicted_Phase"] = [LABEL_MAP[i] for i in y_pred]
    df["True_Phase"] = [LABEL_MAP[i] for i in df["ENSO_Label"]]

    return df, model

def generate_future_features(df, months_ahead=12):
    """Generate simple future predictions using recent trends"""
    last_date = df["Date"].iloc[-1]

    # Get recent trends (last 6 months)
    recent = df.tail(6)

    # Simple linear trend for key variables
    sst_trend = np.polyfit(range(len(recent)), recent["SST_Anomaly"].values, 1)[0]
    soi_trend = np.polyfit(range(len(recent)), recent["SOI"].values, 1)[0]

    future_data = []

    for i in range(1, months_ahead + 1):
        # Calculate future date
        future_date = last_date + pd.DateOffset(months=i)
        month = future_date.month

        # Simple trend projection with seasonal adjustment
        seasonal = 0.2 * np.sin(2 * np.pi * month / 12)
        noise = np.random.normal(0, 0.1)

        sst_forecast = df["SST_Anomaly"].iloc[-1] + sst_trend * i + seasonal + noise
        soi_forecast = df["SOI"].iloc[-1] + soi_trend * i + seasonal + noise

        # Create lag features using recent history or forecasted values
        if i == 1:
            sst_lags = [df["SST_Anomaly"].iloc[-j] for j in range(1, 4)]
            soi_lags = [df["SOI"].iloc[-j] for j in range(1, 4)]
        else:
            # Use previously forecasted values for lags
            prev_forecasts = [item for item in future_data if item]
            sst_lags = []
            soi_lags = []
            for lag in range(1, 4):
                if i - lag <= 0:
                    sst_lags.append(df["SST_Anomaly"].iloc[-(lag-i+1)])
                    soi_lags.append(df["SOI"].iloc[-(lag-i+1)])
                else:
                    sst_lags.append(prev_forecasts[i-lag-1]["SST_Anomaly"])
                    soi_lags.append(prev_forecasts[i-lag-1]["SOI"])

        # Seasonal encoding
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        future_data.append({
            "Date": future_date,
            "SST_Anomaly": sst_forecast,
            "SOI": soi_forecast,
            "SOI_lag_1": soi_lags[0],
            "SOI_lag_2": soi_lags[1] if len(soi_lags) > 1 else soi_lags[0],
            "SOI_lag_3": soi_lags[2] if len(soi_lags) > 2 else soi_lags[0],
            "SST_Anomaly_lag_1": sst_lags[0],
            "SST_Anomaly_lag_2": sst_lags[1] if len(sst_lags) > 1 else sst_lags[0],
            "SST_Anomaly_lag_3": sst_lags[2] if len(sst_lags) > 2 else sst_lags[0],
            "month_sin": month_sin,
            "month_cos": month_cos
        })

    return pd.DataFrame(future_data)

def create_phase_cards():
    """Create educational cards for ENSO phases"""
    descriptions = {
        "La Niña": "Cooler than normal sea surface temperatures. More hurricanes, cooler weather.",
        "Neutral": "Normal sea surface temperatures. Typical seasonal weather patterns.",
        "El Niño": "Warmer than normal sea surface temperatures. Fewer hurricanes, warmer weather."
    }

    st.markdown("### The Three ENSO Phases:")
    cols = st.columns(3)

    for i, (phase, desc) in enumerate(descriptions.items()):
        with cols[i]:
            st.markdown(f"""
            <div style="background: {PHASE_COLORS[phase]}20; padding: 20px; border-radius: 10px; 
                        border-left: 5px solid {PHASE_COLORS[phase]}; height: 150px;">
                <h4 style="color: {PHASE_COLORS[phase]}; margin-top: 0;">{phase}</h4>
                <p style="margin-bottom: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

def show_current_status(df):
    """Display current ENSO conditions"""
    latest = df.iloc[-1]
    date_str = latest["Date"].strftime("%B %Y")

    st.markdown(f"## 📊 Current Status ({date_str})")

    cols = st.columns(4)
    metrics = [
        ("Current Phase", latest["True_Phase"], None),
        ("SST Anomaly", f"{latest['SST_Anomaly']:.2f}°C", None),
        ("ONI Value", f"{latest['ONI']:.2f}", None),
        ("SOI Value", f"{latest['SOI']:.2f}", None)
    ]

    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.metric(label, value, delta)

def create_forecast_plot(df, model, months_ahead):
    """Create forecast visualization"""
    # Generate future predictions
    future_df = generate_future_features(df, months_ahead)
    X_future = future_df[FEATURE_COLS]
    predictions = model.predict(X_future)
    probabilities = model.predict_proba(X_future)

    future_df["Predicted_Phase"] = [LABEL_MAP[i] for i in predictions]
    future_df["Confidence"] = np.max(probabilities, axis=1)

    # Combine recent historical with forecast
    recent_hist = df.tail(24)[["Date", "True_Phase", "ONI"]].copy()
    recent_hist["Type"] = "Historical"

    forecast_data = future_df[["Date", "Predicted_Phase", "SST_Anomaly"]].copy()
    forecast_data.rename(columns={"Predicted_Phase": "True_Phase", "SST_Anomaly": "ONI"}, inplace=True)
    forecast_data["Type"] = "Forecast"

    combined = pd.concat([recent_hist, forecast_data], ignore_index=True)

    # Create plot
    fig = go.Figure()

    for phase in ["La Niña", "Neutral", "El Niño"]:
        # Historical data
        hist_data = combined[(combined["Type"] == "Historical") & (combined["True_Phase"] == phase)]
        if not hist_data.empty:
            fig.add_trace(go.Scatter(
                x=hist_data["Date"], y=hist_data["ONI"],
                mode='markers', name=f"{phase} (Historical)",
                marker=dict(color=PHASE_COLORS[phase], size=8)
            ))

        # Forecast data
        forecast_data = combined[(combined["Type"] == "Forecast") & (combined["True_Phase"] == phase)]
        if not forecast_data.empty:
            fig.add_trace(go.Scatter(
                x=forecast_data["Date"], y=forecast_data["ONI"],
                mode='markers+lines', name=f"{phase} (Forecast)",
                marker=dict(color=PHASE_COLORS[phase], size=10, symbol='diamond'),
                line=dict(color=PHASE_COLORS[phase], dash='dash')
            ))

    # Add threshold lines and current date marker
    fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="El Niño Threshold")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="blue", annotation_text="La Niña Threshold")
    fig.add_vline(x=df["Date"].iloc[-1], line_dash="solid", line_color="black", annotation_text="Now")

    fig.update_layout(
        title="ENSO Forecast: Recent History + Future Predictions",
        xaxis_title="Date", yaxis_title="ENSO Index",
        template="plotly_white", height=500
    )

    return fig, future_df

# Load data
try:
    df, model = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Main App
st.title("🌊 ENSOcast: Climate Pattern Forecasting")
st.markdown("*Understanding and predicting El Niño, La Niña, and Neutral conditions*")

# Sidebar
st.sidebar.title("🧭 Navigation")
pages = ["🎓 Learn", "📊 Current Status", "🔮 Forecast", "📈 History", "⚙️ Model Info"]
page = st.sidebar.selectbox("Go to:", pages)

st.sidebar.markdown("---")
st.sidebar.info(f"Data: {len(df)} months of ENSO observations")

# Page routing
if page == "🎓 Learn":
    st.markdown("## 🌍 What is ENSO?")
    st.markdown("""
    **El Niño-Southern Oscillation (ENSO)** is a climate pattern that affects weather worldwide 
    through changes in Pacific Ocean temperatures and atmospheric pressure.
    """)

    create_phase_cards()

    st.markdown("### 🎯 Why This Matters")
    st.markdown("""
    - **Weather**: Affects global temperature and precipitation patterns
    - **Agriculture**: Influences crop yields and growing conditions  
    - **Disasters**: Changes hurricane frequency and drought/flood risks
    - **Economy**: Impacts fishing, energy, and agricultural markets
    """)

elif page == "📊 Current Status":
    show_current_status(df)

    st.markdown("### 📈 Recent Trend (Last 2 Years)")
    recent = df.tail(24)

    fig = go.Figure()
    for phase in ["La Niña", "Neutral", "El Niño"]:
        phase_data = recent[recent["True_Phase"] == phase]
        if not phase_data.empty:
            fig.add_trace(go.Scatter(
                x=phase_data["Date"], y=phase_data["ONI"],
                mode='markers+lines', name=phase,
                marker=dict(color=PHASE_COLORS[phase], size=8)
            ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="red")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="blue")
    fig.update_layout(title="Recent ENSO Conditions", xaxis_title="Date",
                     yaxis_title="ONI Index", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Forecast":
    st.markdown("## 🔮 ENSO Forecast")

    months_ahead = st.slider("Forecast months ahead:", 3, 24, 12)

    try:
        fig, future_df = create_forecast_plot(df, model, months_ahead)
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        st.markdown("### 📋 Forecast Summary")
        phase_counts = future_df["Predicted_Phase"].value_counts()
        avg_confidence = future_df["Confidence"].mean()

        cols = st.columns(4)
        with cols[0]:
            st.metric("Avg. Confidence", f"{avg_confidence:.1%}")

        for i, (phase, count) in enumerate(phase_counts.items()):
            if i < 3:
                with cols[i+1]:
                    st.metric(f"{phase} Months", f"{count}/{months_ahead}")

        # Detailed forecast table
        st.markdown("### 📅 Monthly Details")
        display_df = future_df[["Date", "Predicted_Phase", "Confidence"]].copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m")
        display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.0%}")
        display_df.columns = ["Month", "Predicted Phase", "Confidence"]
        st.dataframe(display_df, use_container_width=True)

        # Download option
        csv = future_df.to_csv(index=False)
        st.download_button("📥 Download Forecast", csv, "enso_forecast.csv")

    except Exception as e:
        st.error(f"Forecast error: {e}")

    st.warning("⚠️ Predictions are based on historical patterns and may not reflect actual future conditions.")

elif page == "📈 History":
    st.markdown("## 📈 Historical Patterns")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        min_year, max_year = int(df["Date"].dt.year.min()), int(df["Date"].dt.year.max())
        year_range = st.slider("Year range:", min_year, max_year, (max_year-20, max_year))
    with col2:
        phases = st.multiselect("Show phases:", ["La Niña", "Neutral", "El Niño"],
                               default=["La Niña", "Neutral", "El Niño"])

    # Filter data
    filtered = df[
        (df["Date"].dt.year >= year_range[0]) &
        (df["Date"].dt.year <= year_range[1]) &
        (df["True_Phase"].isin(phases))
    ]

    # Timeline plot
    fig = go.Figure()
    for phase in phases:
        phase_data = filtered[filtered["True_Phase"] == phase]
        if not phase_data.empty:
            fig.add_trace(go.Scatter(
                x=phase_data["Date"], y=phase_data["ONI"],
                mode='markers', name=phase,
                marker=dict(color=PHASE_COLORS[phase], size=6)
            ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="red")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="blue")
    fig.update_layout(title="Historical ENSO Timeline", xaxis_title="Date",
                     yaxis_title="ONI Index", template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.markdown("### 📊 Statistics")
    phase_stats = filtered["True_Phase"].value_counts()
    total = len(filtered)

    cols = st.columns(len(phases))
    for i, phase in enumerate(phases):
        count = phase_stats.get(phase, 0)
        pct = (count/total*100) if total > 0 else 0
        with cols[i]:
            st.metric(f"{phase}", f"{count} months", f"{pct:.1f}%")

elif page == "⚙️ Model Info":
    st.markdown("## ⚙️ Model Performance")

    # Accuracy metrics
    y_true, y_pred = df["True_Phase"], df["Predicted_Phase"]
    accuracy = accuracy_score(y_true, y_pred)

    cols = st.columns(3)
    with cols[0]:
        st.metric("Overall Accuracy", f"{accuracy:.1%}")
    with cols[1]:
        st.metric("Data Points", f"{len(df):,}")
    with cols[2]:
        st.metric("Features", len(FEATURE_COLS))

    # Confusion matrix
    st.markdown("### 🎯 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=["La Niña", "Neutral", "El Niño"])
    fig = px.imshow(cm, x=["La Niña", "Neutral", "El Niño"], y=["La Niña", "Neutral", "El Niño"],
                    color_continuous_scale="Blues", text_auto=True,
                    labels=dict(x="Predicted", y="Actual"))
    fig.update_layout(title="Prediction Accuracy by Phase")
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.markdown("### 🔍 Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    # Model details
    st.markdown("### 📋 Model Details")
    st.markdown("""
    - **Type**: Random Forest Classifier
    - **Features**: SST anomalies, SOI values, seasonal patterns, and their lags
    - **Training**: Historical ENSO data from 1982-2025
    - **Purpose**: Educational demonstration of climate pattern prediction
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("*Built by Dylan Dsouza*")
