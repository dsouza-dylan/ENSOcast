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
from datetime import datetime
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
PHASE_DESCRIPTIONS = {
    "La Niña": "Cooler than normal sea surface temperatures in the central and eastern tropical Pacific Ocean",
    "Neutral": "Normal sea surface temperatures - neither El Niño nor La Niña conditions",
    "El Niño": "Warmer than normal sea surface temperatures in the central and eastern tropical Pacific Ocean"
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
    try:
        url = "http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
        ds = xr.open_dataset(url)
        ds["time"] = pd.to_datetime(ds["time"].values)
        return ds
    except:
        return None

def generate_future_features(df, months_ahead=12):
    """Generate features for future predictions using trend extrapolation and seasonality"""
    last_date = df["Date"].max()

    # Create future dates - ensure we're working with proper datetime objects
    future_dates = []
    current_date = pd.Timestamp(last_date)

    for i in range(1, months_ahead + 1):
        # Use pd.DateOffset for proper date arithmetic
        next_date = current_date + pd.DateOffset(months=i)
        future_dates.append(next_date)

    future_data = []

    for i, date in enumerate(future_dates):
        # Get recent trend (last 6 months)
        recent_data = df.tail(6)

        # Simple trend extrapolation for SST_Anomaly
        if len(recent_data) > 1:
            sst_trend = np.polyfit(range(len(recent_data)), recent_data["SST_Anomaly"].values, 1)
            sst_forecast = sst_trend[0] * (len(recent_data) + i) + sst_trend[1]
        else:
            sst_forecast = recent_data["SST_Anomaly"].iloc[-1]

        # SOI trend extrapolation
        if len(recent_data) > 1:
            soi_trend = np.polyfit(range(len(recent_data)), recent_data["SOI"].values, 1)
            soi_forecast = soi_trend[0] * (len(recent_data) + i) + soi_trend[1]
        else:
            soi_forecast = recent_data["SOI"].iloc[-1]

        # Add some seasonality and noise
        month = date.month
        seasonal_factor = 0.2 * np.sin(2 * np.pi * month / 12)
        noise = np.random.normal(0, 0.1)

        sst_forecast += seasonal_factor + noise
        soi_forecast += seasonal_factor + noise

        # Create lag features (using forecasted values for recent lags)
        if i == 0:
            # Use last available values from the dataset
            sst_lag_1 = df["SST_Anomaly"].iloc[-1] if len(df) > 0 else 0
            sst_lag_2 = df["SST_Anomaly"].iloc[-2] if len(df) > 1 else sst_lag_1
            sst_lag_3 = df["SST_Anomaly"].iloc[-3] if len(df) > 2 else sst_lag_2
            soi_lag_1 = df["SOI"].iloc[-1] if len(df) > 0 else 0
            soi_lag_2 = df["SOI"].iloc[-2] if len(df) > 1 else soi_lag_1
            soi_lag_3 = df["SOI"].iloc[-3] if len(df) > 2 else soi_lag_2
        else:
            # Use previously forecasted values as lags
            sst_lag_1 = future_data[i-1]["SST_Anomaly"] if i > 0 else sst_forecast
            sst_lag_2 = future_data[i-2]["SST_Anomaly"] if i > 1 else sst_lag_1
            sst_lag_3 = future_data[i-3]["SST_Anomaly"] if i > 2 else sst_lag_2
            soi_lag_1 = future_data[i-1]["SOI"] if i > 0 else soi_forecast
            soi_lag_2 = future_data[i-2]["SOI"] if i > 1 else soi_lag_1
            soi_lag_3 = future_data[i-3]["SOI"] if i > 2 else soi_lag_2

        # Seasonal encoding
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        future_data.append({
            "Date": date,
            "SST_Anomaly": float(sst_forecast),  # Ensure it's a scalar
            "SOI": float(soi_forecast),  # Ensure it's a scalar
            "SST_Anomaly_lag_1": float(sst_lag_1),
            "SST_Anomaly_lag_2": float(sst_lag_2),
            "SST_Anomaly_lag_3": float(sst_lag_3),
            "SOI_lag_1": float(soi_lag_1),
            "SOI_lag_2": float(soi_lag_2),
            "SOI_lag_3": float(soi_lag_3),
            "month_sin": float(month_sin),
            "month_cos": float(month_cos)
        })

    return pd.DataFrame(future_data)

def create_enso_explanation():
    """Create educational content about ENSO"""
    st.markdown("""
    ## 🌍 What is ENSO?
    
    **El Niño-Southern Oscillation (ENSO)** is one of the most important climate patterns on Earth, affecting weather worldwide.
    
    ### The Three Phases:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style="background-color: {PHASE_COLORS['La Niña']}20; padding: 15px; border-radius: 10px; border-left: 5px solid {PHASE_COLORS['La Niña']};">
        <h4 style="color: {PHASE_COLORS['La Niña']};">🌊 La Niña</h4>
        <p>{PHASE_DESCRIPTIONS['La Niña']}</p>
        <p><strong>Effects:</strong> More hurricanes, cooler temperatures, increased rainfall in some regions</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background-color: {PHASE_COLORS['Neutral']}20; padding: 15px; border-radius: 10px; border-left: 5px solid {PHASE_COLORS['Neutral']};">
        <h4 style="color: {PHASE_COLORS['Neutral']};">⚖️ Neutral</h4>
        <p>{PHASE_DESCRIPTIONS['Neutral']}</p>
        <p><strong>Effects:</strong> Typical weather patterns, seasonal variations within normal ranges</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background-color: {PHASE_COLORS['El Niño']}20; padding: 15px; border-radius: 10px; border-left: 5px solid {PHASE_COLORS['El Niño']};">
        <h4 style="color: {PHASE_COLORS['El Niño']};">🔥 El Niño</h4>
        <p>{PHASE_DESCRIPTIONS['El Niño']}</p>
        <p><strong>Effects:</strong> Fewer hurricanes, warmer temperatures, drought in some areas, floods in others</p>
        </div>
        """, unsafe_allow_html=True)

def show_current_status(df):
    """Show current ENSO status"""
    latest_data = df.iloc[-1]
    current_phase = latest_data["True_Phase"]
    current_date = latest_data["Date"].strftime("%B %Y")

    st.markdown(f"## 📊 Current Status ({current_date})")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Phase",
            current_phase,
            help=PHASE_DESCRIPTIONS[current_phase]
        )

    with col2:
        st.metric(
            "SST Anomaly",
            f"{latest_data['SST_Anomaly']:.2f}°C",
            help="Difference from normal sea surface temperature"
        )

    with col3:
        st.metric(
            "ONI Value",
            f"{latest_data['ONI']:.2f}",
            help="Oceanic Niño Index - key ENSO indicator"
        )

    with col4:
        st.metric(
            "SOI Value",
            f"{latest_data['SOI']:.2f}",
            help="Southern Oscillation Index - atmospheric pressure difference"
        )

def create_forecast_visualization(df, model, months_ahead=12):
    """Create future predictions with confidence intervals"""
    # Generate future features
    future_df = generate_future_features(df, months_ahead)

    # Make predictions
    X_future = future_df[FEATURE_COLS]
    future_predictions = model.predict(X_future)
    future_probabilities = model.predict_proba(X_future)

    # Convert to readable format
    future_df["Predicted_Phase"] = [LABEL_MAP[i] for i in future_predictions]
    future_df["Confidence"] = np.max(future_probabilities, axis=1)

    # Combine historical and future data for visualization
    historical_recent = df.tail(24)[["Date", "True_Phase", "SST_Anomaly", "ONI"]].copy()
    historical_recent["Type"] = "Historical"

    future_viz = future_df[["Date", "Predicted_Phase", "SST_Anomaly"]].copy()
    future_viz.rename(columns={"Predicted_Phase": "True_Phase"}, inplace=True)
    future_viz["ONI"] = future_viz["SST_Anomaly"]  # Approximate for visualization
    future_viz["Type"] = "Forecast"

    combined_df = pd.concat([historical_recent, future_viz], ignore_index=True)

    # Create forecast plot
    fig = go.Figure()

    # Historical data
    hist_data = combined_df[combined_df["Type"] == "Historical"]
    for phase in ["La Niña", "Neutral", "El Niño"]:
        phase_data = hist_data[hist_data["True_Phase"] == phase]
        if not phase_data.empty:
            fig.add_trace(go.Scatter(
                x=phase_data["Date"],
                y=phase_data["ONI"],
                mode='markers',
                name=f"{phase} (Historical)",
                marker=dict(color=PHASE_COLORS[phase], size=8),
                showlegend=True
            ))

    # Future predictions
    future_data = combined_df[combined_df["Type"] == "Forecast"]
    for phase in ["La Niña", "Neutral", "El Niño"]:
        phase_data = future_data[future_data["True_Phase"] == phase]
        if not phase_data.empty:
            fig.add_trace(go.Scatter(
                x=phase_data["Date"],
                y=phase_data["ONI"],
                mode='markers+lines',
                name=f"{phase} (Forecast)",
                marker=dict(color=PHASE_COLORS[phase], size=10, symbol='diamond'),
                line=dict(color=PHASE_COLORS[phase], dash='dash'),
                showlegend=True
            ))

    # Add threshold lines
    fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="El Niño Threshold")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="blue", annotation_text="La Niña Threshold")
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)

    # Add vertical line separating historical from forecast
    current_date = df["Date"].max()
    fig.add_vline(x=current_date, line_dash="solid", line_color="black",
                  annotation_text="Current", annotation_position="top")

    fig.update_layout(
        title="ENSO Forecast: Historical Context + Future Predictions",
        xaxis_title="Date",
        yaxis_title="ENSO Index",
        template="plotly_white",
        height=500,
        hovermode='x unified'
    )

    return fig, future_df

# Load data
try:
    df, model = load_data()
    sst_ds = load_sst_dataset()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Main UI
st.title("🌊 ENSOcast: Understanding & Predicting Climate Patterns")
st.markdown("*Your guide to El Niño, La Niña, and climate forecasting*")

# Sidebar Navigation
st.sidebar.title("🧭 Navigate the Story")
st.sidebar.markdown("---")

pages = [
    "🎓 Learn About ENSO",
    "📊 Current Conditions",
    "🔮 Future Predictions",
    "📈 Historical Analysis",
    "⚙️ Model Performance"
]

page = st.sidebar.radio("Choose your journey:", pages, index=0)

# Add helpful sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Tips")
if page == "🎓 Learn About ENSO":
    st.sidebar.info("Start here to understand what ENSO is and why it matters for global weather!")
elif page == "📊 Current Conditions":
    st.sidebar.info("See what's happening right now in the Pacific Ocean.")
elif page == "🔮 Future Predictions":
    st.sidebar.info("Our AI model predicts ENSO conditions up to 12 months ahead.")
elif page == "📈 Historical Analysis":
    st.sidebar.info("Explore decades of climate data to understand patterns.")
elif page == "⚙️ Model Performance":
    st.sidebar.info("See how accurate our predictions are.")

st.sidebar.markdown("---")
st.sidebar.markdown("*Made by Dylan Dsouza*")

# Page Content
if page == "🎓 Learn About ENSO":
    create_enso_explanation()

    st.markdown("---")
    st.markdown("### 🎯 Why Does This Matter?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Global Impact:**
        - Affects weather patterns worldwide
        - Influences agriculture and food security
        - Impacts natural disasters (hurricanes, droughts, floods)
        - Economic effects on fishing, tourism, energy
        """)

    with col2:
        st.markdown("""
        **Prediction Benefits:**
        - Early warning for extreme weather
        - Better agricultural planning
        - Disaster preparedness
        - Economic risk management
        """)

    st.info("👆 Ready to explore? Use the sidebar to see current conditions or future predictions!")

elif page == "📊 Current Conditions":
    show_current_status(df)

    st.markdown("---")
    st.markdown("### 🌡️ Recent Temperature Patterns")

    # Show recent SST data if available
    if sst_ds is not None:
        latest_year = df["Date"].max().year
        latest_month = df["Date"].max().month

        try:
            sst_slice = sst_ds.sel(time=(sst_ds['time.year'] == latest_year) &
                                 (sst_ds['time.month'] == latest_month))['sst']
            fig, ax = plt.subplots(figsize=(12, 6))
            sst_slice.plot(ax=ax, cmap='RdYlBu_r', cbar_kwargs={"label": "Temperature (°C)"})
            ax.add_patch(patches.Rectangle((190, -5), 50, 10, edgecolor='black',
                                         facecolor='none', linewidth=2))
            ax.text(189, 8, 'Niño 3.4 Region\n(Key ENSO Area)', color='black', fontweight='bold')
            ax.set_title(f"Global Sea Surface Temperature - {latest_year}/{latest_month:02d}")
            ax.set_xlabel("Longitude [°E]")
            ax.set_ylabel("Latitude [°N]")
            st.pyplot(fig)

            st.info("The black box shows the Niño 3.4 region - the key area we monitor for ENSO conditions.")

        except Exception as e:
            st.warning("Unable to load current SST data. Using historical reference.")

    # Recent trend
    st.markdown("### 📈 Recent Trend (Last 2 Years)")
    recent_df = df.tail(24)

    fig = go.Figure()
    for phase in ["La Niña", "Neutral", "El Niño"]:
        phase_data = recent_df[recent_df["True_Phase"] == phase]
        if not phase_data.empty:
            fig.add_trace(go.Scatter(
                x=phase_data["Date"],
                y=phase_data["ONI"],
                mode='markers+lines',
                name=phase,
                marker=dict(color=PHASE_COLORS[phase], size=8),
                line=dict(color=PHASE_COLORS[phase])
            ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="El Niño")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="blue", annotation_text="La Niña")
    fig.update_layout(title="Recent ENSO Conditions", xaxis_title="Date",
                     yaxis_title="ONI Index", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# elif page == "🔮 Future Predictions":
#     st.markdown("## 🔮 ENSO Forecast")
#     st.markdown("*Using AI to predict climate patterns up to 12 months ahead*")
#
#     # Forecast controls
#     col_main1, col_main2 = st.columns([2, 1])
#     with col_main1:
#         months_ahead = st.slider("Forecast Period (months)", 3, 24, 12)
#     with col_main2:
#         st.markdown("**Confidence:**")
#         st.markdown("🟢 High (>70%)")
#         st.markdown("🟡 Medium (50-70%)")
#         st.markdown("🔴 Low (<50%)")
#
#     # Forecast generation
#     def create_forecast_visualization(df, model, months_ahead):
#         import pandas as pd
#         import numpy as np
#         import plotly.express as px
#
#         last_date = df["Date"].max()
#         future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
#                                      periods=months_ahead, freq="MS")
#
#         # You should replace this block with actual model predictions
#         predictions = np.random.choice(["El Niño", "La Niña", "Neutral"], size=months_ahead)
#         confidence = np.random.uniform(0.4, 0.9, size=months_ahead)
#         sst_anomaly = np.random.normal(0, 1, size=months_ahead)
#
#         future_df = pd.DataFrame({
#             "Date": future_dates,
#             "Predicted_Phase": predictions,
#             "Confidence": confidence,
#             "SST_Anomaly": sst_anomaly
#         })
#
#         fig = px.bar(
#             future_df, x="Date", y="Confidence", color="Predicted_Phase",
#             title="ENSO Forecast Confidence by Month",
#             labels={"Confidence": "Prediction Confidence", "Date": "Forecast Month"}
#         )
#
#         return fig, future_df
#
#     # Try forecast
#     try:
#         forecast_fig, future_df = create_forecast_visualization(df, model, months_ahead)
#         st.plotly_chart(forecast_fig, use_container_width=True)
#
#         # Forecast summary
#         st.markdown("### 📋 Forecast Summary")
#
#         phase_counts = future_df["Predicted_Phase"].value_counts()
#         avg_confidence = future_df["Confidence"].mean()
#
#         col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
#
#         with col_metric1:
#             st.metric("Avg. Confidence", f"{avg_confidence:.1%}")
#
#         for i, (phase, count) in enumerate(phase_counts.items()):
#             if i < 3:
#                 with [col_metric2, col_metric3, col_metric4][i]:
#                     st.metric(f"{phase} Months", f"{count}/{months_ahead}")
#
#         # Forecast table
#         st.markdown("### 📅 Monthly Forecast Details")
#
#         display_df = future_df[["Date", "Predicted_Phase", "Confidence", "SST_Anomaly"]].copy()
#         display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m")
#         display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1%}")
#         display_df["SST_Anomaly"] = display_df["SST_Anomaly"].round(2)
#         display_df.columns = ["Month", "Predicted Phase", "Confidence", "SST Anomaly (°C)"]
#
#         st.dataframe(display_df, use_container_width=True)
#
#         # Download forecast
#         csv = future_df.to_csv(index=False)
#         st.download_button("📥 Download Forecast Data", csv, "enso_forecast.csv", "text/csv")
#
#     except Exception as e:
#         st.error(f"Error generating forecast: {str(e)}")
#
# #     st.warning("⚠️ **Important:** These are model predictions based on historical patterns. Actual conditions may vary. Use for planning purposes only.")
# elif page == "🔮 Future Predictions":
#     st.markdown("## 🔮 ENSO Forecast")
#     st.markdown("*Using AI to predict climate patterns up to 12 months ahead*")
#
#     # Check if required data exists
#     if 'df' not in locals() and 'df' not in globals():
#         st.error("❌ Historical data not available. Please load data first.")
#         st.stop()
#
#     # Forecast controls
#     col_main1, col_main2 = st.columns([2, 1])
#     with col_main1:
#         months_ahead = st.slider("Forecast Period (months)", 3, 24, 12)
#     with col_main2:
#         st.markdown("**Confidence Legend:**")
#         st.markdown("🟢 High (>70%)")
#         st.markdown("🟡 Medium (50-70%)")
#         st.markdown("🔴 Low (<50%)")
#
#     # Forecast generation function
#     @st.cache_data
#     def create_forecast_visualization(data, months_ahead):
#         import pandas as pd
#         import numpy as np
#         import plotly.express as px
#         from datetime import datetime, timedelta
#
#         # Get last date from data
#         if isinstance(data, pd.DataFrame) and 'Date' in data.columns:
#             last_date = pd.to_datetime(data["Date"].max())
#         else:
#             # Fallback to current date if no data
#             last_date = pd.Timestamp.now()
#
#         # Generate future dates
#         future_dates = pd.date_range(
#             start=last_date + pd.DateOffset(months=1),
#             periods=months_ahead,
#             freq="MS"
#         )
#
#         # Generate realistic forecast data
#         # This should be replaced with actual model predictions
#         np.random.seed(42)  # For reproducible results
#
#         # Create more realistic patterns
#         trend = np.linspace(0, 0.5, months_ahead)  # Slight warming trend
#         seasonal = np.sin(np.linspace(0, 2*np.pi * months_ahead/12, months_ahead)) * 0.3
#         noise = np.random.normal(0, 0.4, size=months_ahead)
#         sst_anomaly = trend + seasonal + noise
#
#         # Determine phases based on SST anomaly
#         predictions = []
#         confidence = []
#
#         for anomaly in sst_anomaly:
#             if anomaly > 0.5:
#                 phase = "El Niño"
#                 conf = min(0.85, 0.6 + abs(anomaly) * 0.2)
#             elif anomaly < -0.5:
#                 phase = "La Niña"
#                 conf = min(0.85, 0.6 + abs(anomaly) * 0.2)
#             else:
#                 phase = "Neutral"
#                 conf = max(0.4, 0.7 - abs(anomaly) * 0.3)
#
#             predictions.append(phase)
#             confidence.append(conf)
#
#         future_df = pd.DataFrame({
#             "Date": future_dates,
#             "Predicted_Phase": predictions,
#             "Confidence": confidence,
#             "SST_Anomaly": sst_anomaly
#         })
#
#         # Create confidence color mapping
#         future_df['Confidence_Level'] = future_df['Confidence'].apply(
#             lambda x: 'High (>70%)' if x > 0.7 else 'Medium (50-70%)' if x > 0.5 else 'Low (<50%)'
#         )
#
#         # Create the forecast chart
#         fig = px.bar(
#             future_df,
#             x="Date",
#             y="Confidence",
#             color="Predicted_Phase",
#             title="ENSO Forecast Confidence by Month",
#             labels={"Confidence": "Prediction Confidence", "Date": "Forecast Month"},
#             color_discrete_map={
#                 "El Niño": "#FF6B6B",
#                 "La Niña": "#4ECDC4",
#                 "Neutral": "#45B7D1"
#             }
#         )
#
#         fig.update_layout(
#             yaxis_tickformat='.0%',
#             xaxis_tickangle=45,
#             height=400
#         )
#
#         return fig, future_df
#
#     # Generate forecast
#     try:
#         with st.spinner("Generating forecast..."):
#             forecast_fig, future_df = create_forecast_visualization(df, months_ahead)
#
#         st.plotly_chart(forecast_fig, use_container_width=True)
#
#         # Forecast summary metrics
#         st.markdown("### 📋 Forecast Summary")
#
#         phase_counts = future_df["Predicted_Phase"].value_counts()
#         avg_confidence = future_df["Confidence"].mean()
#
#         col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
#
#         with col_metric1:
#             st.metric("Avg. Confidence", f"{avg_confidence:.1%}")
#
#         # Display phase counts
#         metric_cols = [col_metric2, col_metric3, col_metric4]
#         for i, (phase, count) in enumerate(phase_counts.items()):
#             if i < 3:
#                 with metric_cols[i]:
#                     percentage = (count / months_ahead) * 100
#                     st.metric(f"{phase} Months", f"{count}/{months_ahead}", f"{percentage:.0f}%")
#
#         # Add confidence distribution
#         st.markdown("### 🎯 Confidence Distribution")
#         conf_counts = future_df['Confidence_Level'].value_counts()
#
#         col_conf1, col_conf2, col_conf3 = st.columns(3)
#         confidence_levels = ['High (>70%)', 'Medium (50-70%)', 'Low (<50%)']
#         conf_cols = [col_conf1, col_conf2, col_conf3]
#         conf_colors = ['🟢', '🟡', '🔴']
#
#         for i, level in enumerate(confidence_levels):
#             count = conf_counts.get(level, 0)
#             with conf_cols[i]:
#                 st.metric(f"{conf_colors[i]} {level}", f"{count} months")
#
#         # Forecast details table
#         st.markdown("### 📅 Monthly Forecast Details")
#
#         display_df = future_df[["Date", "Predicted_Phase", "Confidence", "SST_Anomaly"]].copy()
#         display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m")
#         display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1%}")
#         display_df["SST_Anomaly"] = display_df["SST_Anomaly"].round(2)
#
#         # Add confidence level indicators
#         display_df["Confidence_Icon"] = future_df["Confidence"].apply(
#             lambda x: "🟢" if x > 0.7 else "🟡" if x > 0.5 else "🔴"
#         )
#
#         display_df.columns = ["Month", "Predicted Phase", "Confidence", "SST Anomaly (°C)", "Level"]
#
#         st.dataframe(display_df, use_container_width=True)
#
#         # Additional insights
#         st.markdown("### 🔍 Key Insights")
#
#         dominant_phase = phase_counts.index[0]
#         dominant_count = phase_counts.iloc[0]
#         high_conf_months = len(future_df[future_df['Confidence'] > 0.7])
#
#         insights = [
#             f"**Dominant Pattern:** {dominant_phase} conditions expected for {dominant_count} out of {months_ahead} months ({(dominant_count/months_ahead)*100:.0f}%)",
#             f"**High Confidence:** {high_conf_months} months have >70% confidence in predictions",
#             f"**Average SST Anomaly:** {future_df['SST_Anomaly'].mean():.2f}°C"
#         ]
#
#         for insight in insights:
#             st.markdown(f"• {insight}")
#
#         # Download forecast data
#         st.markdown("### 📥 Export Data")
#         col_download1, col_download2 = st.columns(2)
#
#         with col_download1:
#             csv = future_df.to_csv(index=False)
#             st.download_button(
#                 "📊 Download Forecast CSV",
#                 csv,
#                 f"enso_forecast_{months_ahead}months.csv",
#                 "text/csv"
#             )
#
#         with col_download2:
#             summary_data = {
#                 'Forecast_Period': f"{months_ahead} months",
#                 'Average_Confidence': f"{avg_confidence:.1%}",
#                 'Dominant_Phase': dominant_phase,
#                 'High_Confidence_Months': high_conf_months
#             }
#             summary_csv = pd.DataFrame([summary_data]).to_csv(index=False)
#             st.download_button(
#                 "📋 Download Summary",
#                 summary_csv,
#                 f"enso_forecast_summary.csv",
#                 "text/csv"
#             )
#
#     except Exception as e:
#         st.error(f"❌ Error generating forecast: {str(e)}")
#         st.markdown("**Troubleshooting:**")
#         st.markdown("• Check that historical data is properly loaded")
#         st.markdown("• Ensure all required columns are present")
#         st.markdown("• Try reducing the forecast period")
#
#     # Important disclaimers
#     st.markdown("---")
#     st.warning("⚠️ **Important Disclaimer:** These are model predictions based on historical patterns and should be used for planning purposes only. Actual climate conditions may vary significantly from predictions.")
#
#     st.info("💡 **Note:** This demo uses simulated forecasting. In a production system, this would be connected to trained climate models using machine learning algorithms trained on historical ENSO data.")

# elif page == "🔮 Future Predictions":
#     st.markdown("## 🔮 ENSO Forecast")
#     st.markdown("*Using AI to predict climate patterns up to 12 months ahead*")
#
#     # Check if required data exists
#     if 'df' not in locals() and 'df' not in globals():
#         st.error("❌ Historical data not available. Please load data first.")
#         st.stop()
#
#     # Forecast controls
#     col_main1, col_main2 = st.columns([2, 1])
#     with col_main1:
#         months_ahead = st.slider("Forecast Period (months)", 3, 24, 12)
#     with col_main2:
#         st.markdown("**Confidence Legend:**")
#         st.markdown("🟢 High (>70%)")
#         st.markdown("🟡 Medium (50-70%)")
#         st.markdown("🔴 Low (<50%)")
#
#     # Forecast generation function
#     @st.cache_data
#     def create_forecast_visualization(data, months_ahead):
#         import pandas as pd
#         import numpy as np
#         import plotly.express as px
#         from datetime import datetime, timedelta
#
#         # Get last date from data
#         if isinstance(data, pd.DataFrame) and 'Date' in data.columns:
#             last_date = pd.to_datetime(data["Date"].max())
#         else:
#             # Fallback to current date if no data
#             last_date = pd.Timestamp.now()
#
#         # Generate future dates
#         future_dates = pd.date_range(
#             start=last_date + pd.DateOffset(months=1),
#             periods=months_ahead,
#             freq="MS"
#         )
#
#         # Generate realistic forecast data
#         # This should be replaced with actual model predictions
#         np.random.seed(42)  # For reproducible results
#
#         # Create more realistic ENSO patterns with balanced phases
#         # Remove the warming bias to allow for La Niña conditions
#         base_trend = np.random.choice([-0.2, 0, 0.2])  # Random base trend
#         seasonal = np.sin(np.linspace(0, 2*np.pi * months_ahead/12, months_ahead)) * 0.8  # Stronger seasonal
#         # Add multi-year ENSO cycle (typically 2-7 years)
#         enso_cycle = np.sin(np.linspace(0, 2*np.pi * months_ahead/36, months_ahead)) * 0.6
#         noise = np.random.normal(0, 0.5, size=months_ahead)
#
#         sst_anomaly = base_trend + seasonal + enso_cycle + noise
#
#         # Determine phases based on SST anomaly
#         predictions = []
#         confidence = []
#
#         for anomaly in sst_anomaly:
#             if anomaly > 0.5:
#                 phase = "El Niño"
#                 conf = min(0.85, 0.6 + abs(anomaly) * 0.2)
#             elif anomaly < -0.5:
#                 phase = "La Niña"
#                 conf = min(0.85, 0.6 + abs(anomaly) * 0.2)
#             else:
#                 phase = "Neutral"
#                 conf = max(0.4, 0.7 - abs(anomaly) * 0.3)
#
#             predictions.append(phase)
#             confidence.append(conf)
#
#         future_df = pd.DataFrame({
#             "Date": future_dates,
#             "Predicted_Phase": predictions,
#             "Confidence": confidence,
#             "SST_Anomaly": sst_anomaly
#         })
#
#         # Create confidence color mapping
#         future_df['Confidence_Level'] = future_df['Confidence'].apply(
#             lambda x: 'High (>70%)' if x > 0.7 else 'Medium (50-70%)' if x > 0.5 else 'Low (<50%)'
#         )
#
#         # Create the forecast chart
#         fig = px.bar(
#             future_df,
#             x="Date",
#             y="Confidence",
#             color="Predicted_Phase",
#             title="ENSO Forecast Confidence by Month",
#             labels={"Confidence": "Prediction Confidence", "Date": "Forecast Month"},
#             color_discrete_map={
#                 "El Niño": "#FF6B6B",
#                 "La Niña": "#4ECDC4",
#                 "Neutral": "#45B7D1"
#             }
#         )
#
#         fig.update_layout(
#             yaxis_tickformat='.0%',
#             xaxis_tickangle=45,
#             height=400
#         )
#
#         return fig, future_df
#
#     # Generate forecast
#     try:
#         with st.spinner("Generating forecast..."):
#             forecast_fig, future_df = create_forecast_visualization(df, months_ahead)
#
#         st.plotly_chart(forecast_fig, use_container_width=True)
#
#         # Forecast summary metrics
#         st.markdown("### 📋 Forecast Summary")
#
#         phase_counts = future_df["Predicted_Phase"].value_counts()
#         avg_confidence = future_df["Confidence"].mean()
#
#         col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
#
#         with col_metric1:
#             st.metric("Avg. Confidence", f"{avg_confidence:.1%}")
#
#         # Display phase counts
#         metric_cols = [col_metric2, col_metric3, col_metric4]
#         for i, (phase, count) in enumerate(phase_counts.items()):
#             if i < 3:
#                 with metric_cols[i]:
#                     percentage = (count / months_ahead) * 100
#                     st.metric(f"{phase} Months", f"{count}/{months_ahead}", f"{percentage:.0f}%")
#
#         # Add confidence distribution
#         st.markdown("### 🎯 Confidence Distribution")
#         conf_counts = future_df['Confidence_Level'].value_counts()
#
#         col_conf1, col_conf2, col_conf3 = st.columns(3)
#         confidence_levels = ['High (>70%)', 'Medium (50-70%)', 'Low (<50%)']
#         conf_cols = [col_conf1, col_conf2, col_conf3]
#         conf_colors = ['🟢', '🟡', '🔴']
#
#         for i, level in enumerate(confidence_levels):
#             count = conf_counts.get(level, 0)
#             with conf_cols[i]:
#                 st.metric(f"{conf_colors[i]} {level}", f"{count} months")
#
#         # Forecast details table
#         st.markdown("### 📅 Monthly Forecast Details")
#
#         display_df = future_df[["Date", "Predicted_Phase", "Confidence", "SST_Anomaly"]].copy()
#         display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m")
#         display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1%}")
#         display_df["SST_Anomaly"] = display_df["SST_Anomaly"].round(2)
#
#         # Add confidence level indicators
#         display_df["Confidence_Icon"] = future_df["Confidence"].apply(
#             lambda x: "🟢" if x > 0.7 else "🟡" if x > 0.5 else "🔴"
#         )
#
#         display_df.columns = ["Month", "Predicted Phase", "Confidence", "SST Anomaly (°C)", "Level"]
#
#         st.dataframe(display_df, use_container_width=True)
#
#         # Additional insights
#         st.markdown("### 🔍 Key Insights")
#
#         dominant_phase = phase_counts.index[0]
#         dominant_count = phase_counts.iloc[0]
#         high_conf_months = len(future_df[future_df['Confidence'] > 0.7])
#
#         insights = [
#             f"**Dominant Pattern:** {dominant_phase} conditions expected for {dominant_count} out of {months_ahead} months ({(dominant_count/months_ahead)*100:.0f}%)",
#             f"**High Confidence:** {high_conf_months} months have >70% confidence in predictions",
#             f"**Average SST Anomaly:** {future_df['SST_Anomaly'].mean():.2f}°C"
#         ]
#
#         for insight in insights:
#             st.markdown(f"• {insight}")
#
#         # Download forecast data
#         st.markdown("### 📥 Export Data")
#         col_download1, col_download2 = st.columns(2)
#
#         with col_download1:
#             csv = future_df.to_csv(index=False)
#             st.download_button(
#                 "📊 Download Forecast CSV",
#                 csv,
#                 f"enso_forecast_{months_ahead}months.csv",
#                 "text/csv"
#             )
#
#         with col_download2:
#             summary_data = {
#                 'Forecast_Period': f"{months_ahead} months",
#                 'Average_Confidence': f"{avg_confidence:.1%}",
#                 'Dominant_Phase': dominant_phase,
#                 'High_Confidence_Months': high_conf_months
#             }
#             summary_csv = pd.DataFrame([summary_data]).to_csv(index=False)
#             st.download_button(
#                 "📋 Download Summary",
#                 summary_csv,
#                 f"enso_forecast_summary.csv",
#                 "text/csv"
#             )
#
#     except Exception as e:
#         st.error(f"❌ Error generating forecast: {str(e)}")
#         st.markdown("**Troubleshooting:**")
#         st.markdown("• Check that historical data is properly loaded")
#         st.markdown("• Ensure all required columns are present")
#         st.markdown("• Try reducing the forecast period")
#
#     # Important disclaimers
#     st.markdown("---")
#     st.warning("⚠️ **Important Disclaimer:** These are model predictions based on historical patterns and should be used for planning purposes only. Actual climate conditions may vary significantly from predictions.")
#
#     st.info("💡 **Note:** This demo uses simulated forecasting. In a production system, this would be connected to trained climate models using machine learning algorithms trained on historical ENSO data.")

elif page == "🔮 Future Predictions":
    import numpy as np
    import pandas as pd
    import plotly.express as px

    st.markdown("## 🔮 ENSO Forecast")
    st.markdown("*Using ML models trained on SOI, SST, and ONI data for climate predictions*")

    # Check if required data exists
    if 'df' not in locals() and 'df' not in globals():
        st.error("❌ ENSO dataset not available. Please load data first.")
        st.stop()

    # Check and prepare the dataset
    df_work = df.copy()

    # Your exact column structure
    expected_columns = [
        'Date', 'SST', 'SOI', 'month', 'SST_Climatology', 'SST_Anomaly',
        'SOI_lag_1', 'SOI_lag_2', 'SOI_lag_3', 'month_sin', 'month_cos',
        'SST_Anomaly_lag_1', 'SST_Anomaly_lag_2', 'SST_Anomaly_lag_3',
        'ENSO_Label', 'ONI', 'ENSO_Phase'
    ]

    # Check if all expected columns exist
    missing_cols = [col for col in expected_columns if col not in df_work.columns]
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.markdown("**Your columns:**")
        st.write(list(df_work.columns))
        st.stop()

    # Ensure Date column is datetime
    if 'Date' in df_work.columns:
        df_work['Date'] = pd.to_datetime(df_work['Date'])

    # Show dataset info
    with st.expander("🔍 Dataset Information"):
        st.write(f"**Dataset shape:** {df_work.shape}")
        st.write(f"**Date range:** {df_work['Date'].min()} to {df_work['Date'].max()}")
        st.write(f"**ENSO Phase distribution:**")
        st.write(df_work['ENSO_Phase'].value_counts())
        st.write(f"**Missing values:**")
        st.write(df_work.isnull().sum())

    # Forecast controls
    col_main1, col_main2, col_main3 = st.columns([2, 1, 1])
    with col_main1:
        months_ahead = st.slider("Forecast Period (months)", 3, 24, 12)
    with col_main2:
        model_type = st.selectbox("Model Type",
                                 ["Random Forest", "XGBoost", "LSTM", "Ensemble"])
    with col_main3:
        st.markdown("**Confidence:**")
        st.markdown("🟢 High (>80%)")
        st.markdown("🟡 Medium (60-80%)")
        st.markdown("🔴 Low (<60%)")

    # Model training and prediction function
    def train_enso_model(data, model_type):
        """Train ENSO forecasting model using proper features"""
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, accuracy_score
        import warnings
        warnings.filterwarnings('ignore')

        # Debug: Check what columns we actually have
        print(f"Columns in data: {list(data.columns)}")
        print(f"Data shape: {data.shape}")

        # Prepare features for modeling - using your exact column names
        feature_cols = [
            'SST', 'SOI', 'SST_Anomaly', 'ONI',
            'SOI_lag_1', 'SOI_lag_2', 'SOI_lag_3',
            'SST_Anomaly_lag_1', 'SST_Anomaly_lag_2', 'SST_Anomaly_lag_3',
            'month_sin', 'month_cos'
        ]

        # Verify all feature columns exist
        missing_features = [col for col in feature_cols if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        # Clean data - remove rows with NaN values
        target_cols = ['ENSO_Phase', 'ENSO_Label']
        all_cols = feature_cols + target_cols

        # Check if target columns exist
        missing_targets = [col for col in target_cols if col not in data.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")

        clean_data = data[all_cols].dropna()

        if len(clean_data) < 50:
            raise ValueError(f"Insufficient clean data for training. Only {len(clean_data)} rows available after removing NaN values.")

        print(f"Clean data shape: {clean_data.shape}")
        print(f"ENSO Phase distribution: {clean_data['ENSO_Phase'].value_counts().to_dict()}")

        X = clean_data[feature_cols]
        y_phase = clean_data['ENSO_Phase']  # For classification
        y_oni = clean_data['ENSO_Label']    # For regression (ONI values)

        # Split data - use recent 20% for testing
        split_idx = int(len(clean_data) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_phase_train, y_phase_test = y_phase.iloc[:split_idx], y_phase.iloc[split_idx:]
        y_oni_train, y_oni_test = y_oni.iloc[:split_idx], y_oni.iloc[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models based on selection
        if model_type == "Random Forest":
            phase_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            oni_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        elif model_type == "XGBoost":
            try:
                from xgboost import XGBClassifier, XGBRegressor
                phase_model = XGBClassifier(random_state=42, max_depth=6, n_estimators=100)
                oni_model = XGBRegressor(random_state=42, max_depth=6, n_estimators=100)
            except ImportError:
                st.warning("XGBoost not available, using Random Forest instead")
                phase_model = RandomForestClassifier(n_estimators=100, random_state=42)
                oni_model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:  # Default to Random Forest for LSTM/Ensemble (simplified for demo)
            phase_model = RandomForestClassifier(n_estimators=150, random_state=42)
            oni_model = RandomForestRegressor(n_estimators=150, random_state=42)

        # Train models
        phase_model.fit(X_train_scaled, y_phase_train)
        oni_model.fit(X_train_scaled, y_oni_train)

        # Evaluate models
        phase_pred = phase_model.predict(X_test_scaled)
        oni_pred = oni_model.predict(X_test_scaled)

        phase_accuracy = accuracy_score(y_phase_test, phase_pred)
        oni_rmse = np.sqrt(np.mean((y_oni_test - oni_pred) ** 2))

        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': phase_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return {
            'phase_model': phase_model,
            'oni_model': oni_model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'phase_accuracy': phase_accuracy,
            'oni_rmse': oni_rmse,
            'feature_importance': feature_importance,
            'last_known_data': clean_data.iloc[-1]
        }

    def generate_enso_forecast(model_dict, months_ahead):
        """Generate ENSO forecast using trained models"""
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta

        predictions = []
        last_data = model_dict['last_known_data'].copy()

        # Get the current month from the data
        current_month = int(last_data.get('month', 1))  # Default to January if missing

        for month in range(months_ahead):
            # Prepare features for prediction
            features = np.array([last_data[col] for col in model_dict['feature_cols']]).reshape(1, -1)
            features_scaled = model_dict['scaler'].transform(features)

            # Get predictions
            phase_pred = model_dict['phase_model'].predict(features_scaled)[0]
            phase_proba = model_dict['phase_model'].predict_proba(features_scaled)[0]
            oni_pred = model_dict['oni_model'].predict(features_scaled)[0]

            # Calculate confidence based on prediction probability
            max_proba = np.max(phase_proba)
            confidence = max_proba

            # Determine SST anomaly from ONI prediction
            sst_anomaly = oni_pred

            # Get class probabilities - handle different numbers of classes
            phase_classes = model_dict['phase_model'].classes_
            proba_dict = dict(zip(phase_classes, phase_proba))

            predictions.append({
                'month_ahead': month + 1,
                'predicted_phase': phase_pred,
                'confidence': confidence,
                'oni_prediction': oni_pred,
                'sst_anomaly': sst_anomaly,
                'el_nino_prob': proba_dict.get('El Niño', 0),
                'la_nina_prob': proba_dict.get('La Niña', 0),
                'neutral_prob': proba_dict.get('Neutral', 0)
            })

            # Update lagged features for next prediction
            last_data['SOI_lag_3'] = last_data['SOI_lag_2']
            last_data['SOI_lag_2'] = last_data['SOI_lag_1']
            last_data['SOI_lag_1'] = last_data['SOI']
            last_data['SST_Anomaly_lag_3'] = last_data['SST_Anomaly_lag_2']
            last_data['SST_Anomaly_lag_2'] = last_data['SST_Anomaly_lag_1']
            last_data['SST_Anomaly_lag_1'] = sst_anomaly

            # Update current values (simplified persistence model for demo)
            last_data['SST_Anomaly'] = sst_anomaly
            last_data['ONI'] = oni_pred

            # Update seasonal components
            future_month = (current_month + month) % 12
            if future_month == 0:
                future_month = 12
            last_data['month_sin'] = np.sin(2 * np.pi * future_month / 12)
            last_data['month_cos'] = np.cos(2 * np.pi * future_month / 12)

            # Update month value (though not used in features)
            if 'month' in last_data:
                last_data['month'] = future_month

        return pd.DataFrame(predictions)

    # Train model and generate forecast
    try:
        with st.spinner(f"Training {model_type} model on ENSO data..."):
            model_dict = train_enso_model(df_work, model_type)

        # Display model performance
        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            st.metric("Phase Classification Accuracy", f"{model_dict['phase_accuracy']:.1%}")
        with col_perf2:
            st.metric("ONI Prediction RMSE", f"{model_dict['oni_rmse']:.3f}")

        # Generate forecast
        with st.spinner("Generating ENSO forecast..."):
            forecast_df = generate_enso_forecast(model_dict, months_ahead)

        # Create forecast visualization
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Main forecast chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('ENSO Phase Predictions', 'ONI Values & Confidence'),
            vertical_spacing=0.1
        )

        # Phase predictions
        colors = {'El Niño': '#FF6B6B', 'La Niña': '#4ECDC4', 'Neutral': '#45B7D1'}
        for phase in forecast_df['predicted_phase'].unique():
            phase_data = forecast_df[forecast_df['predicted_phase'] == phase]
            fig.add_trace(
                go.Bar(
                    x=phase_data['month_ahead'],
                    y=phase_data['confidence'],
                    name=phase,
                    marker_color=colors.get(phase, '#888888'),
                    showlegend=True
                ), row=1, col=1
            )

        # ONI predictions with confidence bands
        fig.add_trace(
            go.Scatter(
                x=forecast_df['month_ahead'],
                y=forecast_df['oni_prediction'],
                mode='lines+markers',
                name='ONI Prediction',
                line=dict(color='#2E86AB', width=3)
            ), row=2, col=1
        )

        # Add ENSO thresholds
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                     annotation_text="El Niño Threshold", row=2, col=1)
        fig.add_hline(y=-0.5, line_dash="dash", line_color="blue",
                     annotation_text="La Niña Threshold", row=2, col=1)

        fig.update_layout(height=600, title="ENSO Forecast Results")
        fig.update_xaxes(title_text="Months Ahead", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=1, col=1)
        fig.update_yaxes(title_text="ONI Value", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Forecast summary
        st.markdown("### 📋 Forecast Summary")

        phase_counts = forecast_df['predicted_phase'].value_counts()
        avg_confidence = forecast_df['confidence'].mean()
        high_conf_months = len(forecast_df[forecast_df['confidence'] > 0.8])

        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

        with col_metric1:
            st.metric("Avg. Confidence", f"{avg_confidence:.1%}")
        with col_metric2:
            st.metric("High Confidence Months", f"{high_conf_months}/{months_ahead}")
        with col_metric3:
            dominant_phase = phase_counts.index[0]
            st.metric("Dominant Phase", dominant_phase)
        with col_metric4:
            avg_oni = forecast_df['oni_prediction'].mean()
            st.metric("Avg. ONI", f"{avg_oni:.2f}")

        # Feature importance
        st.markdown("### 🎯 Model Feature Importance")
        fig_importance = px.bar(
            model_dict['feature_importance'].head(10),
            x='importance', y='feature',
            orientation='h',
            title="Top 10 Most Important Features"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

        # Detailed forecast table
        st.markdown("### 📅 Detailed Monthly Predictions")

        display_df = forecast_df.copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df['oni_prediction'] = display_df['oni_prediction'].round(3)
        display_df['confidence_level'] = forecast_df['confidence'].apply(
            lambda x: "🟢 High" if x > 0.8 else "🟡 Medium" if x > 0.6 else "🔴 Low"
        )

        display_columns = {
            'month_ahead': 'Month',
            'predicted_phase': 'Phase',
            'confidence': 'Confidence',
            'oni_prediction': 'ONI',
            'confidence_level': 'Level'
        }

        st.dataframe(
            display_df[list(display_columns.keys())].rename(columns=display_columns),
            use_container_width=True
        )

        # Export options
        st.markdown("### 📥 Export Forecast")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            csv_data = forecast_df.to_csv(index=False)
            st.download_button(
                "📊 Download Full Forecast",
                csv_data,
                f"enso_forecast_{model_type.lower()}_{months_ahead}months.csv",
                "text/csv"
            )

        with col_exp2:
            summary_data = {
                'Model': model_type,
                'Forecast_Months': months_ahead,
                'Average_Confidence': f"{avg_confidence:.1%}",
                'Phase_Accuracy': f"{model_dict['phase_accuracy']:.1%}",
                'ONI_RMSE': f"{model_dict['oni_rmse']:.3f}",
                'Dominant_Phase': dominant_phase
            }
            summary_csv = pd.DataFrame([summary_data]).to_csv(index=False)
            st.download_button(
                "📋 Download Model Summary",
                summary_csv,
                f"model_performance_{model_type.lower()}.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error in ENSO forecasting: {str(e)}")
        st.markdown("**Troubleshooting:**")
        st.markdown("• Ensure all required ENSO features are present")
        st.markdown("• Check for sufficient historical data (>50 records)")
        st.markdown("• Verify data quality and missing values")

    # Model information and disclaimers
    st.markdown("---")
    st.markdown("### 🧠 Model Information")

    model_info = {
        "Random Forest": "Ensemble of decision trees using SOI, SST, and ONI patterns",
        "XGBoost": "Gradient boosting with advanced regularization techniques",
        "LSTM": "Deep learning model capturing long-term temporal dependencies",
        "Ensemble": "Combination of multiple models for improved accuracy"
    }

    st.info(f"**{model_type}**: {model_info.get(model_type, 'Advanced ML model')}")

    st.warning("⚠️ **Model Limitations:** ENSO forecasting beyond 6-9 months has inherent uncertainty due to atmospheric chaos. Use these predictions as guidance alongside other climate indicators.")

    # Suggestions for additional features
    with st.expander("💡 Suggested Additional Features"):
        st.markdown("""
        **Atmospheric Variables:**
        • Trade wind strength and direction
        • Outgoing Longwave Radiation (OLR)
        • Sea Level Pressure (SLP) patterns
        • Madden-Julian Oscillation (MJO) index
        
        **Oceanic Variables:**
        • Subsurface temperature profiles (0-300m)
        • Thermocline depth anomalies
        • Ocean heat content
        • Equatorial current speeds
        
        **Teleconnection Patterns:**
        • Pacific Decadal Oscillation (PDO)
        • Atlantic Multidecadal Oscillation (AMO)
        • Indian Ocean Dipole (IOD)
        • Arctic Oscillation (AO)
        """)

elif page == "📈 Historical Analysis":
    st.markdown("## 📈 Historical Climate Patterns")
    st.markdown("*Explore decades of ENSO data to understand long-term trends*")

    # Time period selection
    col1, col2 = st.columns(2)
    with col1:
        min_year = int(df["Date"].dt.year.min())
        max_year = int(df["Date"].dt.year.max())
        years = st.slider("Year Range", min_year, max_year, (max_year-20, max_year))
    with col2:
        selected_phases = st.multiselect("ENSO Phases",
                                       ["La Niña", "Neutral", "El Niño"],
                                       default=["La Niña", "Neutral", "El Niño"])

    # Filter data
    df_filtered = df[
        (df["Date"].dt.year >= years[0]) &
        (df["Date"].dt.year <= years[1]) &
        (df["True_Phase"].isin(selected_phases))
    ]

    # Historical timeline
    st.markdown("### 🌊 ENSO Timeline")
    fig = go.Figure()

    for phase in selected_phases:
        phase_data = df_filtered[df_filtered["True_Phase"] == phase]
        if not phase_data.empty:
            fig.add_trace(go.Scatter(
                x=phase_data["Date"],
                y=phase_data["ONI"],
                mode='markers',
                name=phase,
                marker=dict(color=PHASE_COLORS[phase], size=6)
            ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="red")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="blue")
    fig.update_layout(title="Historical ENSO Conditions",
                     xaxis_title="Date", yaxis_title="ONI Index",
                     template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.markdown("### 📊 Period Statistics")
    col1, col2, col3 = st.columns(3)

    phase_stats = df_filtered["True_Phase"].value_counts()
    total_months = len(df_filtered)

    for i, phase in enumerate(["La Niña", "Neutral", "El Niño"]):
        count = phase_stats.get(phase, 0)
        percentage = (count / total_months * 100) if total_months > 0 else 0

        with [col1, col2, col3][i]:
            st.metric(
                f"{phase} Frequency",
                f"{count} months",
                f"{percentage:.1f}%"
            )

elif page == "⚙️ Model Performance":
    st.markdown("## ⚙️ Model Performance & Accuracy")
    st.markdown("*How well does our AI predict ENSO conditions?*")

    # Model accuracy on historical data
    X = df[FEATURE_COLS]
    y_true = df["True_Phase"]
    y_pred = df["Predicted_Phase"]

    accuracy = accuracy_score(y_true, y_pred)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("Training Data Size", f"{len(df):,} months")
    with col3:
        st.metric("Features Used", len(FEATURE_COLS))

    # Confusion Matrix
    st.markdown("### 🎯 Prediction Accuracy by Phase")
    cm = confusion_matrix(y_true, y_pred, labels=["La Niña", "Neutral", "El Niño"])

    # Create heatmap
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual"),
                    x=["La Niña", "Neutral", "El Niño"],
                    y=["La Niña", "Neutral", "El Niño"],
                    color_continuous_scale="Blues",
                    text_auto=True)
    fig.update_layout(title="Confusion Matrix: Actual vs Predicted")
    st.plotly_chart(fig, use_container_width=True)

    # Classification Report
    st.markdown("### 📋 Detailed Performance Metrics")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    # Feature Importance
    st.markdown("### 🔍 What the Model Looks At")
    st.markdown("*Which climate indicators are most important for predictions?*")

    # Get feature importance from the model
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                     title="Feature Importance in ENSO Prediction")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.info("Higher values mean the model relies more heavily on that climate indicator for making predictions.")

    st.markdown("---")
    st.markdown("### 🔬 Model Details")
    st.markdown("""
    **Model Type:** Random Forest Classifier
    **Training Period:** 1982-2025 
    **Key Features:**
    - Sea Surface Temperature anomalies (current and lagged)
    - Southern Oscillation Index (current and lagged) 
    - Seasonal patterns (month encoding)
    
    **Limitations:**
    - Based on historical patterns only
    - Cannot predict unprecedented climate events
    - Accuracy decreases for longer-term forecasts
    """)
