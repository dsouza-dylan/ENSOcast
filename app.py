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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP (optional dependency)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Page config with custom styling
st.set_page_config(
    page_title="ENSOcast - Climate Storytelling",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced aesthetics
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .story-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .phase-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .el-nino { background: linear-gradient(135deg, #ff9a8b 0%, #f6416c 100%); color: white; }
    .la-nina { background: linear-gradient(135deg, #a8edea 0%, #3b82f6 100%); color: white; }
    .neutral { background: linear-gradient(135deg, #d299c2 0%, #fef9d3 100%); color: #333; }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .insight-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1rem 0;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_data():
    try:
        df = pd.read_csv("merged_enso.csv", parse_dates=["Date"])
        model = joblib.load("enso_model_a_baseline.pkl")
        return df, model
    except FileNotFoundError:
        st.error("Data files not found. Please ensure merged_enso.csv and enso_model_a_baseline.pkl are in the correct directory.")
        return None, None

@st.cache_data
def load_sst_dataset():
    try:
        url = "http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
        ds = xr.open_dataset(url)
        ds["time"] = pd.to_datetime(ds["time"].values)
        return ds
    except:
        st.warning("Could not connect to live SST data. Using cached data for demonstration.")
        return None

def create_story_intro():
    st.markdown("""
    <div class="main-header">
        <h1>🌊 ENSOcast</h1>
        <h3>Decoding Earth's Most Powerful Climate Pattern</h3>
        <p>Journey through the invisible forces that shape weather across our planet</p>
    </div>
    """, unsafe_allow_html=True)

def explain_enso_story():
    st.markdown("""
    <div class="story-card">
        <h2>🌍 The Story of ENSO</h2>
        <p>Imagine the Pacific Ocean as Earth's heartbeat. Every few years, this heartbeat changes rhythm, 
        sending ripples of change across continents. This is ENSO - the El Niño-Southern Oscillation.</p>
        
        <p>🔄 <strong>It's a conversation between ocean and atmosphere:</strong></p>
        <ul>
            <li>The ocean warms or cools</li>
            <li>The atmosphere responds by shifting winds</li>
            <li>These winds push ocean currents in new directions</li>
            <li>Weather patterns worldwide transform</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def create_phase_cards():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="phase-card el-nino">
            <h3>🔴 El Niño</h3>
            <p><strong>"The Little Boy"</strong></p>
            <p>Ocean warms up<br>
            Brings floods to some,<br>
            droughts to others</p>
            <small>Occurs every 2-7 years</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="phase-card neutral">
            <h3>⚪ Neutral</h3>
            <p><strong>"The Quiet Phase"</strong></p>
            <p>Ocean at normal temps<br>
            Weather patterns<br>
            follow usual seasons</p>
            <small>Most common state</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="phase-card la-nina">
            <h3>🔵 La Niña</h3>
            <p><strong>"The Little Girl"</strong></p>
            <p>Ocean cools down<br>
            Intensifies hurricanes,<br>
            brings extreme weather</p>
            <small>Often follows El Niño</small>
        </div>
        """, unsafe_allow_html=True)

def create_feature_for_date(target_date, df, feature_cols):
    """Create features for a specific date based on historical patterns"""
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    elif isinstance(target_date, datetime.date):
        target_date = pd.Timestamp(target_date)

    df_sorted = df.sort_values('Date')
    time_diffs = (df_sorted['Date'] - target_date).dt.days.abs()
    closest_idx = time_diffs.idxmin()
    closest_position = df_sorted.index.get_loc(closest_idx)

    month = target_date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    start_idx = max(0, closest_position - 12)
    end_idx = closest_position + 1
    recent_data = df_sorted.iloc[start_idx:end_idx]

    if len(recent_data) == 0:
        feature_values = df[feature_cols].mean()
    else:
        feature_values = recent_data[feature_cols].mean()

    if 'month_sin' in feature_cols:
        feature_values['month_sin'] = month_sin
    if 'month_cos' in feature_cols:
        feature_values['month_cos'] = month_cos

    return feature_values.values.reshape(1, -1)

# Load data with error handling
data_loaded = True
try:
    df, model = load_model_and_data()
    if df is None or model is None:
        data_loaded = False
    else:
        sst_ds = load_sst_dataset()

        feature_cols = [
            "SST_Anomaly", "SOI", "SOI_lag_1", "SOI_lag_2", "SOI_lag_3",
            "SST_Anomaly_lag_1", "SST_Anomaly_lag_2", "SST_Anomaly_lag_3",
            "month_sin", "month_cos"
        ]
        X = df[feature_cols]
        y_true = df["ENSO_Label"]
        y_pred = model.predict(X)

        label_map = {0: "La Niña", 1: "Neutral", 2: "El Niño"}
        df["Predicted_Phase"] = [label_map[i] for i in y_pred]
        df["True_Phase"] = [label_map[i] for i in y_true]
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

# Create the storytelling interface
create_story_intro()

if not data_loaded:
    st.error("⚠️ Unable to load required data files. Please check your setup.")
    st.stop()

# Sidebar with narrative navigation
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
           padding: 1rem; border-radius: 10px; color: white; text-align: center;">
    <h3>🧭 Your Journey</h3>
    <p>Navigate through the ENSO story</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation with storytelling approach
page = st.sidebar.radio(
    "Choose Your Path:",
    [
        "🌟 Start Here: Understanding ENSO",
        "🔮 The Oracle: Make Predictions",
        "📊 The Evidence: Historical Patterns",
        "🌡️ The Global View: Ocean Temperatures",
        "🔬 Behind the Scenes: Model Performance",
        "🛠️ Experiment: Train Your Own Model"
    ],
    index=0
)

if page == "🌟 Start Here: Understanding ENSO":
    explain_enso_story()

    st.markdown("### 🎭 Meet the Three Characters")
    create_phase_cards()

    st.markdown("""
    <div class="story-card">
        <h3>🎯 Why Does This Matter?</h3>
        <p>ENSO affects:</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="background: #ff6b6b; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>🌾 Agriculture</h4>
                <p>Crop yields, drought, floods</p>
            </div>
            <div style="background: #4ecdc4; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>🌪️ Weather</h4>
                <p>Hurricanes, storms, rainfall</p>
            </div>
            <div style="background: #45b7d1; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>💰 Economy</h4>
                <p>Energy costs, insurance, trade</p>
            </div>
            <div style="background: #f9ca24; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>🌊 Marine Life</h4>
                <p>Fish populations, coral health</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Recent ENSO timeline
    st.markdown("### 📅 ENSO's Recent Journey")
    recent_data = df[df["Date"] >= (df["Date"].max() - pd.DateOffset(years=5))].copy()

    fig = px.line(recent_data, x="Date", y="ONI",
                  title="Oceanic Niño Index - The Last 5 Years",
                  color_discrete_sequence=['#667eea'])

    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="El Niño Threshold", annotation_position="top right")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue",
                  annotation_text="La Niña Threshold", annotation_position="bottom right")

    # Add shaded regions
    fig.add_hrect(y0=0.5, y1=3, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=-3, y1=-0.5, fillcolor="blue", opacity=0.1, line_width=0)

    fig.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <p>📈 <strong>Reading the story:</strong> When the line goes above +0.5, it's El Niño territory (red zone). 
        Below -0.5 means La Niña conditions (blue zone). The line tells the story of ocean temperature changes!</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "🔮 The Oracle: Make Predictions":
    st.markdown("""
    <div class="story-card">
        <h2>🔮 The ENSO Oracle</h2>
        <p>Step into the role of a climate prophet. Our AI has studied decades of ocean and atmospheric data 
        to peer into the future. What will the Pacific Ocean whisper about the months ahead?</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced prediction interface
    st.markdown("### 🎯 Cast Your Prediction")

    col1, col2 = st.columns([1, 1])
    with col1:
        target_year = st.number_input("🗓️ Which year calls to you?",
                                     min_value=1982, max_value=2030, value=2024,
                                     help="Choose any year from 1982 to 2030")

    with col2:
        target_month = st.selectbox("🌙 Which month holds the mystery?", [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ], index=datetime.datetime.now().month-1)

    predict_button = st.button("🔮 Reveal the Future", type="primary", use_container_width=True)

    if predict_button:
        month_num = {
            "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
            "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
        }[target_month]

        target_date = datetime.date(target_year, month_num, 1)

        with st.spinner("🌊 The ocean spirits are consulting... Reading the signs..."):
            try:
                X_target = create_feature_for_date(target_date, df, feature_cols)
                prediction = model.predict(X_target)[0]
                probabilities = model.predict_proba(X_target)[0]

                predicted_phase = label_map[prediction]
                max_prob = max(probabilities)

                # Dramatic reveal
                st.balloons()

                if predicted_phase == "El Niño":
                    st.markdown(f"""
                    <div class="prediction-box" style="background: linear-gradient(135deg, #ff9a8b 0%, #f6416c 100%);">
                        <h1>🔴 El Niño Awakens</h1>
                        <h3>For {target_month} {target_year}</h3>
                        <p style="font-size: 1.2em;">The ocean will run warm with El Niño's fire. 
                        Expect the unexpected - flooding rains in some lands, drought in others.</p>
                        <p><strong>Oracle's Confidence:</strong> {max_prob:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                elif predicted_phase == "La Niña":
                    st.markdown(f"""
                    <div class="prediction-box" style="background: linear-gradient(135deg, #a8edea 0%, #3b82f6 100%);">
                        <h1>🔵 La Niña's Cool Embrace</h1>
                        <h3>For {target_month} {target_year}</h3>
                        <p style="font-size: 1.2em;">The ocean will run cold under La Niña's influence. 
                        Hurricanes may dance with greater fury, and weather patterns will intensify.</p>
                        <p><strong>Oracle's Confidence:</strong> {max_prob:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div class="prediction-box" style="background: linear-gradient(135deg, #d299c2 0%, #fef9d3 100%); color: #333;">
                        <h1>⚪ The Neutral Path</h1>
                        <h3>For {target_month} {target_year}</h3>
                        <p style="font-size: 1.2em;">The ocean rests in balance. Weather patterns will follow 
                        their seasonal rhythms without dramatic shifts.</p>
                        <p><strong>Oracle's Confidence:</strong> {max_prob:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### 🎲 The Full Prophecy")

                # Probability visualization
                prob_data = pd.DataFrame({
                    'Phase': ['🔵 La Niña', '⚪ Neutral', '🔴 El Niño'],
                    'Probability': probabilities,
                    'Colors': ['#3b82f6', '#94a3b8', '#f6416c']
                })

                fig = px.bar(prob_data, x='Phase', y='Probability',
                           color='Colors', color_discrete_map='identity',
                           title="The Oracle's Vision - Detailed Probabilities")
                fig.update_layout(showlegend=False, template="plotly_white")
                fig.update_yaxis(title="Probability", tickformat='.0%')

                st.plotly_chart(fig, use_container_width=True)

                # Confidence interpretation
                if max_prob > 0.8:
                    confidence_story = "🎯 **Crystal Clear Vision** - The signs are unmistakable"
                elif max_prob > 0.6:
                    confidence_story = "👁️ **Strong Intuition** - The patterns point clearly in one direction"
                elif max_prob > 0.4:
                    confidence_story = "🤔 **Clouded Vision** - The future remains uncertain, multiple paths possible"
                else:
                    confidence_story = "🌫️ **Misty Prophecy** - The ocean spirits are conflicted"

                st.markdown(f"""
                <div class="insight-card">
                    {confidence_story}
                    <br><br>
                    <em>"The further we peer into time's river, the murkier the waters become. 
                    Use this wisdom as a guide, not gospel."</em>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ The oracle's vision is clouded: {e}")

elif page == "📊 The Evidence: Historical Patterns":
    st.markdown("""
    <div class="story-card">
        <h2>📊 Chronicles of the Past</h2>
        <p>Every climate prediction is built on the foundation of history. Let's explore the patterns 
        hidden in decades of data - the rhythm of ENSO through time.</p>
    </div>
    """, unsafe_allow_html=True)

    # Interactive year selection with story
    st.markdown("### 🕰️ Choose Your Era")
    years = st.slider("Explore ENSO history across decades", 1982, 2024, (2000, 2020),
                     help="Drag to select the time period you want to explore")

    selected_phases = st.multiselect(
        "Which characters in the ENSO story interest you?",
        ["La Niña", "Neutral", "El Niño"],
        default=["La Niña", "Neutral", "El Niño"],
        help="Select which ENSO phases to include in your analysis"
    )

    # Filter data
    df_filtered = df[
        (df["Date"].dt.year >= years[0]) &
        (df["Date"].dt.year <= years[1]) &
        (df["ENSO_Phase"].isin(selected_phases))
    ]

    if len(df_filtered) == 0:
        st.warning("No data available for your selected criteria. Try adjusting your filters.")
        st.stop()

    # Key statistics in narrative form
    phase_counts = df_filtered["ENSO_Phase"].value_counts()
    total_months = len(df_filtered)
    years_span = years[1] - years[0] + 1

    st.markdown(f"""
    <div class="story-card">
        <h3>📈 Your Selected Era: {years[0]} - {years[1]}</h3>
        <p>In these <strong>{years_span} years</strong> ({total_months} months), here's how ENSO spent its time:</p>
    </div>
    """, unsafe_allow_html=True)

    # Phase distribution with enhanced visuals
    col1, col2, col3 = st.columns(3)
    for i, (phase, count) in enumerate(phase_counts.items()):
        percentage = (count / total_months) * 100
        color = "#f6416c" if phase == "El Niño" else "#3b82f6" if phase == "La Niña" else "#94a3b8"
        emoji = "🔴" if phase == "El Niño" else "🔵" if phase == "La Niña" else "⚪"

        if i == 0:
            col = col1
        elif i == 1:
            col = col2
        else:
            col = col3

        with col:
            st.markdown(f"""
            <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40);">
                <h2>{emoji} {phase}</h2>
                <h3>{count} months</h3>
                <p>{percentage:.1f}% of the time</p>
            </div>
            """, unsafe_allow_html=True)

    # Enhanced timeline visualizations
    st.markdown("### 🌊 The Ocean's Temperature Story")

    # SST Timeline with narrative
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["SST"],
            name="Ocean Temperature",
            line=dict(color='#ff6b6b', width=2),
            hovertemplate="<b>%{x}</b><br>Temperature: %{y:.2f}°C<extra></extra>"
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["SST_Climatology"],
            name="Expected Temperature",
            line=dict(color='#4ecdc4', dash='dot', width=2),
            hovertemplate="<b>%{x}</b><br>Expected: %{y:.2f}°C<extra></extra>"
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title="The Pacific's Temperature Dance",
        xaxis_title="Time",
        template="plotly_white",
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Sea Surface Temperature (°C)", secondary_y=False)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <p>🌡️ <strong>Temperature tells the tale:</strong> When the red line rises above the blue dotted line, 
        El Niño is stirring. When it falls below, La Niña is taking hold.</p>
    </div>
    """, unsafe_allow_html=True)

    # ENSO Phase Timeline
    st.markdown("### 🎭 The Three-Act Drama")

    # Create phase timeline
    phase_numeric = df_filtered["ENSO_Phase"].map({"La Niña": -1, "Neutral": 0, "El Niño": 1})

    fig_phases = go.Figure()

    # Color mapping for phases
    colors = {"La Niña": "#3b82f6", "Neutral": "#94a3b8", "El Niño": "#f6416c"}

    for phase in df_filtered["ENSO_Phase"].unique():
        phase_data = df_filtered[df_filtered["ENSO_Phase"] == phase]
        phase_values = phase_data["ENSO_Phase"].map({"La Niña": -1, "Neutral": 0, "El Niño": 1})

        fig_phases.add_trace(go.Scatter(
            x=phase_data["Date"],
            y=phase_values,
            mode='markers',
            name=f"{phase}",
            marker=dict(color=colors[phase], size=8, opacity=0.7),
            hovertemplate=f"<b>{phase}</b><br>%{{x}}<extra></extra>"
        ))

    fig_phases.update_layout(
        title="ENSO's Dramatic Timeline",
        xaxis_title="Time",
        yaxis=dict(
            tickmode='array',
            tickvals=[-1, 0, 1],
            ticktext=['🔵 La Niña', '⚪ Neutral', '🔴 El Niño']
        ),
        template="plotly_white",
        height=300,
        showlegend=True
    )

    st.plotly_chart(fig_phases, use_container_width=True)

    # Advanced pattern analysis
    st.markdown("### 🔍 Hidden Patterns Revealed")

    # Seasonal analysis
    df_filtered['Month'] = df_filtered['Date'].dt.month
    seasonal_patterns = df_filtered.groupby(['Month', 'ENSO_Phase']).size().unstack(fill_value=0)

    fig_seasonal = px.bar(
        seasonal_patterns.reset_index().melt(id_vars='Month', var_name='Phase', value_name='Count'),
        x='Month', y='Count', color='Phase',
        title="When Do Different ENSO Phases Prefer to Appear?",
        color_discrete_map=colors
    )

    fig_seasonal.update_xaxes(
        tickmode='array',
        tickvals=list(range(1, 13)),
        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )

    st.plotly_chart(fig_seasonal, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <p>📅 <strong>Seasonal Secrets:</strong> ENSO phases have favorite seasons! Notice how some phases 
        appear more often in certain months - this is one of the patterns our prediction model learned.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "🌡️ The Global View: Ocean Temperatures":
    st.markdown("""
    <div class="story-card">
        <h2>🌡️ The Global Ocean's Portrait</h2>
        <p>Witness the Pacific Ocean as seen from space - a living, breathing entity whose temperature patterns 
        tell the story of global climate. The Niño 3.4 region (our black box) is ENSO's heartbeat.</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced date selection
    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.slider("🗓️ Journey through time", min_value=1982, max_value=2024, value=2010)
    with col2:
        month_dict = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        selected_month = st.selectbox("🌙 Choose your window into the ocean",
                                    list(month_dict.keys()), index=7)

    month_num = month_dict[selected_month]

    # Historical context for selected date
    if selected_year >= 2015:
        era_story = "🌊 **Modern Era** - Recent climate patterns with enhanced monitoring"
    elif selected_year >= 2000:
        era_story = "📡 **Satellite Age** - High-resolution ocean observations"
    elif selected_year >= 1990:
        era_story = "🔬 **Scientific Revolution** - ENSO understanding deepened"
    else:
        era_story = "🗿 **Early Records** - Foundation of climate science"

    st.markdown(f"""
    <div class="insight-card">
        <h4>{selected_month} {selected_year}</h4>
        <p>{era_story}</p>
    </div>
    """, unsafe_allow_html=True)

    # SST visualization with enhanced storytelling
    if sst_ds is not None:
        try:
            with st.spinner(f"🛰️ Downloading satellite data for {selected_month} {selected_year}..."):
                sst_slice = sst_ds.sel(time=(sst_ds['time.year'] == selected_year) &
                                     (sst_ds['time.month'] == month_num))['sst']

                fig, ax = plt.subplots(figsize=(15, 8))

                # Enhanced colormap and styling
                im = sst_slice.plot(ax=ax, cmap='RdYlBu_r',
                                  cbar_kwargs={"label": "Sea Surface Temperature (°C)", "shrink": 0.8})

                # Highlight Niño 3.4 region with enhanced styling
                nino_rect = patches.Rectangle((190, -5), 50, 10,
                                            edgecolor='black', facecolor='none',
                                            linewidth=3, linestyle='--')
                ax.add_patch(nino_rect)

                # Add annotations
                ax.text(215, 8, '🎯 Niño 3.4 Region\n(ENSO\'s Heartbeat)',
                       ha='center', va='bottom', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

                ax.set_title(f'🌊 Pacific Ocean Temperature Portrait - {selected_month} {selected_year}',
                           fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel("Longitude (°E)", fontsize=12)
                ax.set_ylabel("Latitude (°N)", fontsize=12)

                # Add temperature context
                avg_temp = float(sst_slice.mean())
                max_temp = float(sst_slice.max())
                min_temp = float(sst_slice.min())

                st.pyplot(fig)

                # Temperature insights
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>🌡️ Average</h3>
                        <h2>{avg_temp:.1f}°C</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>🔥 Hottest</h3>
                        <h2>{max_temp:.1f}°C</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>🧊 Coolest</h3>
                        <h2>{min_temp:.1f}°C</h2>
                    </div>
                    """, unsafe_allow_html=True)

                # ENSO context for this specific date
                date_context = df[df['Date'].dt.year == selected_year]
                if len(date_context) > 0:
                    monthly_data = date_context[date_context['Date'].dt.month == month_num]
                    if len(monthly_data) > 0:
                        phase = monthly_data.iloc[0]['ENSO_Phase']
                        oni_value = monthly_data.iloc[0]['ONI']

                        phase_colors = {"El Niño": "#f6416c", "La Niña": "#3b82f6", "Neutral": "#94a3b8"}
                        phase_emojis = {"El Niño": "🔴", "La Niña": "🔵", "Neutral": "⚪"}

                        st.markdown(f"""
                        <div class="insight-card" style="background: {phase_colors[phase]}20;">
                            <h3>{phase_emojis[phase]} {selected_month} {selected_year} was a <strong>{phase}</strong> month</h3>
                            <p>ONI Value: <strong>{oni_value:.2f}</strong></p>
                            <p>This ocean temperature pattern was {'typical' if phase == 'Neutral' else 'influenced by ' + phase + ' conditions'}.</p>
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"🌊 Unable to load ocean data for {selected_month} {selected_year}: {e}")
            st.info("💡 Try selecting a different date, or check your internet connection for satellite data.")
    else:
        st.warning("🛰️ Live satellite data unavailable. Here's what you would see:")
        st.markdown("""
        <div class="story-card">
            <h3>🌊 Ocean Temperature Visualization</h3>
            <p>Normally, you'd see a colorful map of the Pacific Ocean where:</p>
            <ul>
                <li>🔴 <strong>Red/Yellow areas</strong> show warmer waters (El Niño influence)</li>
                <li>🔵 <strong>Blue areas</strong> show cooler waters (La Niña influence)</li>
                <li>🎯 <strong>The black box</strong> highlights the Niño 3.4 region - ENSO's control center</li>
                <li>🌡️ <strong>Temperature gradients</strong> reveal the ocean's story</li>
            </ul>
            <p>This satellite view helps scientists understand how ENSO affects global weather patterns.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔬 Behind the Scenes: Model Performance":
    st.markdown("""
    <div class="story-card">
        <h2>🔬 The Science Behind the Oracle</h2>
        <p>Every prediction has a story of how it was made. Let's peek behind the curtain at our AI model - 
        its successes, its struggles, and what makes it tick.</p>
    </div>
    """, unsafe_allow_html=True)

    # Model performance metrics with storytelling
    accuracy = accuracy_score(df["True_Phase"], df["Predicted_Phase"])

    st.markdown("### 🎯 The Report Card")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🎯 Overall Accuracy</h3>
            <h2>{accuracy:.0%}</h2>
            <p>Correct predictions</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        total_predictions = len(df)
        correct_predictions = int(accuracy * total_predictions)
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Total Tested</h3>
            <h2>{total_predictions:,}</h2>
            <p>Months analyzed</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>✅ Success Stories</h3>
            <h2>{correct_predictions:,}</h2>
            <p>Months predicted correctly</p>
        </div>
        """, unsafe_allow_html=True)

    # Performance story
    if accuracy > 0.8:
        performance_story = "🌟 **Excellent Performance** - Our model is a skilled climate detective!"
    elif accuracy > 0.7:
        performance_story = "👍 **Good Performance** - Reliable predictions with room for improvement"
    elif accuracy > 0.6:
        performance_story = "🤔 **Moderate Performance** - Better than guessing, but climate is complex"
    else:
        performance_story = "🔧 **Needs Improvement** - Climate prediction remains challenging"

    st.markdown(f"""
    <div class="insight-card">
        <h3>{performance_story}</h3>
        <p>Out of {total_predictions:,} months of historical data, our AI correctly identified the ENSO phase 
        {correct_predictions:,} times. That means it was right {accuracy:.0%} of the time!</p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed performance breakdown
    st.markdown("### 📋 The Detailed Scorecard")

    # Classification report in a more narrative format
    report = classification_report(df["True_Phase"], df["Predicted_Phase"], output_dict=True)

    phases = ["La Niña", "Neutral", "El Niño"]
    phase_emojis = {"La Niña": "🔵", "Neutral": "⚪", "El Niño": "🔴"}

    for phase in phases:
        if phase in report:
            precision = report[phase]['precision']
            recall = report[phase]['recall']
            f1 = report[phase]['f1-score']
            support = int(report[phase]['support'])

            st.markdown(f"""
            <div class="insight-card">
                <h4>{phase_emojis[phase]} {phase} Performance</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                    <div>
                        <strong>Precision:</strong> {precision:.0%}<br>
                        <small>When model says "{phase}", it's right {precision:.0%} of the time</small>
                    </div>
                    <div>
                        <strong>Recall:</strong> {recall:.0%}<br>
                        <small>Model catches {recall:.0%} of actual {phase} events</small>
                    </div>
                    <div>
                        <strong>F1-Score:</strong> {f1:.0%}<br>
                        <small>Balanced performance measure</small>
                    </div>
                    <div>
                        <strong>Sample Size:</strong> {support}<br>
                        <small>Months of {phase} in our data</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Confusion matrix as a story
    st.markdown("### 🤷 Where the Model Gets Confused")

    cm = confusion_matrix(df["True_Phase"], df["Predicted_Phase"], labels=["La Niña", "Neutral", "El Niño"])

    # Create a heatmap-style visualization
    fig_cm = px.imshow(cm,
                       labels=dict(x="Predicted Phase", y="Actual Phase", color="Count"),
                       x=["🔵 La Niña", "⚪ Neutral", "🔴 El Niño"],
                       y=["🔵 La Niña", "⚪ Neutral", "🔴 El Niño"],
                       color_continuous_scale="Blues",
                       title="Confusion Matrix: Where Predictions Go Wrong")

    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            fig_cm.add_annotation(x=j, y=i, text=str(cm[i][j]),
                                showarrow=False, font_size=16, font_color="white" if cm[i][j] > cm.max()/2 else "black")

    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <p>📊 <strong>Reading the confusion:</strong> The diagonal shows correct predictions (darker = better). 
        Off-diagonal squares show mistakes - when the model confused one phase for another.</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance story
    st.markdown("### 🔍 What the Model Pays Attention To")

    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        # Create feature importance with descriptions
        feature_descriptions = {
            "SST_Anomaly": "🌡️ Current ocean temperature difference",
            "SOI": "🌬️ Current atmospheric pressure pattern",
            "SOI_lag_1": "🌬️ Last month's atmospheric pressure",
            "SOI_lag_2": "🌬️ Two months ago atmospheric pressure",
            "SOI_lag_3": "🌬️ Three months ago atmospheric pressure",
            "SST_Anomaly_lag_1": "🌡️ Last month's ocean temperature",
            "SST_Anomaly_lag_2": "🌡️ Two months ago ocean temperature",
            "SST_Anomaly_lag_3": "🌡️ Three months ago ocean temperature",
            "month_sin": "📅 Seasonal pattern (sine)",
            "month_cos": "📅 Seasonal pattern (cosine)"
        }

        fig_importance = px.bar(importance_df, y="Feature", x="Importance",
                              orientation="h", title="The Model's Decision Factors")
        fig_importance.update_layout(height=500)

        st.plotly_chart(fig_importance, use_container_width=True)

        # Top 3 features explanation
        top_features = importance_df.tail(3)
        st.markdown("### 🏆 The Top 3 Decision Makers")

        for i, (_, row) in enumerate(top_features.iterrows()):
            feature = row['Feature']
            importance = row['Importance']
            rank = ["🥉 Third", "🥈 Second", "🥇 First"][i]

            st.markdown(f"""
            <div class="insight-card">
                <h4>{rank} Most Important: {feature_descriptions.get(feature, feature)}</h4>
                <p>Influence Score: <strong>{importance:.1%}</strong></p>
                <p>This factor accounts for {importance:.1%} of the model's decision-making process.</p>
            </div>
            """, unsafe_allow_html=True)

    # Model limitations and honesty
    st.markdown("### 🚧 The Model's Limitations")
    st.markdown("""
    <div class="story-card">
        <h4>🤖 What Our AI Can and Cannot Do</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
            <div>
                <h5>✅ Strengths</h5>
                <ul>
                    <li>Learns from decades of data</li>
                    <li>Considers multiple climate factors</li>
                    <li>Provides probability estimates</li>
                    <li>Fast and consistent predictions</li>
                </ul>
            </div>
            <div>
                <h5>⚠️ Limitations</h5>
                <ul>
                    <li>Climate is inherently unpredictable</li>
                    <li>Rare events are hard to forecast</li>
                    <li>Models can't capture everything</li>
                    <li>Accuracy decreases with time</li>
                </ul>
            </div>
        </div>
        <p><em>💡 Remember: Even the best climate models are tools to help us understand probabilities, not crystal balls that guarantee the future.</em></p>
    </div>
    """, unsafe_allow_html=True)

elif page == "🛠️ Experiment: Train Your Own Model":
    st.markdown("""
    <div class="story-card">
        <h2>🛠️ Become a Climate AI Trainer</h2>
        <p>Ready to train your own ENSO prediction model? Choose your data, pick your algorithm, 
        and see how different approaches perform. This is where science meets experimentation!</p>
    </div>
    """, unsafe_allow_html=True)

    # Interactive model building
    st.markdown("### 🎛️ Design Your Experiment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📅 Choose Your Training Period")
        years = st.slider("Select years for training", 1982, 2024, (2000, 2020),
                         help="More years = more data to learn from")

        st.markdown("#### 🎭 Focus on Specific Phases")
        selected_phases = st.multiselect(
            "Which ENSO phases to include?",
            ["La Niña", "Neutral", "El Niño"],
            default=["La Niña", "Neutral", "El Niño"],
            help="You can focus on specific phases to see how models handle them"
        )

    with col2:
        st.markdown("#### 🤖 Choose Your AI Algorithm")
        classifier_options = {
            "Random Forest": "🌳 Uses many decision trees - good for complex patterns",
            "Support Vector Machine": "📐 Finds optimal boundaries - good for clear separations",
            "Logistic Regression": "📊 Simple linear approach - fast and interpretable"
        }

        selected_classifier = st.selectbox(
            "Pick your AI algorithm:",
            list(classifier_options.keys()),
            help="Each algorithm has different strengths"
        )

        st.markdown(f"""
        <div class="insight-card">
            <p>{classifier_options[selected_classifier]}</p>
        </div>
        """, unsafe_allow_html=True)

    # Train button
    if st.button("🚀 Train Your Model", type="primary", use_container_width=True):

        # Filter data based on selections
        filtered_df = df[
            (df["Date"].dt.year >= years[0]) &
            (df["Date"].dt.year <= years[1]) &
            (df["True_Phase"].isin(selected_phases))
        ]

        if len(filtered_df) < 50:
            st.warning("⚠️ Not enough data for reliable training. Try expanding your date range or including more phases.")
            st.stop()

        with st.spinner("🧠 Training your AI model... Teaching it to recognize ENSO patterns..."):
            X_custom = filtered_df[feature_cols]
            y_custom = filtered_df["True_Phase"]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_custom, y_custom, test_size=0.3, shuffle=False, random_state=42
            )

            # Train selected model
            models = {
                "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
                "Support Vector Machine": SVC(random_state=42, probability=True),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
            }

            custom_model = models[selected_classifier]
            custom_model.fit(X_train, y_train)
            y_pred_custom = custom_model.predict(X_test)

            # Calculate performance
            custom_accuracy = accuracy_score(y_test, y_pred_custom)

            # Results presentation
            st.balloons()

            st.markdown("### 🎉 Your Model is Ready!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>🎯 Accuracy</h3>
                    <h2>{custom_accuracy:.0%}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                training_size = len(X_train)
                st.markdown(f"""
                <div class="metric-container">
                    <h3>📚 Training Data</h3>
                    <h2>{training_size}</h2>
                    <small>months</small>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                test_size = len(X_test)
                st.markdown(f"""
                <div class="metric-container">
                    <h3>🧪 Test Data</h3>
                    <h2>{test_size}</h2>
                    <small>months</small>
                </div>
                """, unsafe_allow_html=True)

            # Performance comparison
            baseline_accuracy = accuracy_score(df["True_Phase"], df["Predicted_Phase"])

            if custom_accuracy > baseline_accuracy:
                comparison = f"🎉 **Outstanding!** Your model ({custom_accuracy:.0%}) outperformed our baseline ({baseline_accuracy:.0%})"
                comparison_color = "green"
            elif custom_accuracy > baseline_accuracy - 0.05:
                comparison = f"👍 **Good work!** Your model ({custom_accuracy:.0%}) performed similarly to our baseline ({baseline_accuracy:.0%})"
                comparison_color = "blue"
            else:
                comparison = f"🤔 **Learning opportunity!** Your model ({custom_accuracy:.0%}) has room for improvement vs baseline ({baseline_accuracy:.0%})"
                comparison_color = "orange"

            st.markdown(f"""
            <div class="insight-card" style="border-left-color: {comparison_color};">
                <h4>📊 Performance Comparison</h4>
                <p>{comparison}</p>
            </div>
            """, unsafe_allow_html=True)

            # Detailed results
            st.markdown("### 📋 Detailed Performance Report")

            # Per-phase performance
            report = classification_report(y_test, y_pred_custom, output_dict=True)

            phase_emojis = {"La Niña": "🔵", "Neutral": "⚪", "El Niño": "🔴"}

            for phase in selected_phases:
                if phase in report:
                    precision = report[phase]['precision']
                    recall = report[phase]['recall']
                    f1 = report[phase]['f1-score']
                    support = int(report[phase]['support'])

                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>{phase_emojis[phase]} {phase} Results</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem;">
                            <div><strong>Precision:</strong> {precision:.0%}</div>
                            <div><strong>Recall:</strong> {recall:.0%}</div>
                            <div><strong>F1-Score:</strong> {f1:.0%}</div>
                            <div><strong>Test Cases:</strong> {support}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Feature importance (if available)
            if hasattr(custom_model, 'feature_importances_'):
                st.markdown("### 🔍 What Your Model Learned to Focus On")

                importance_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": custom_model.feature_importances_
                }).sort_values("Importance", ascending=True)

                fig = px.bar(importance_df, y="Feature", x="Importance",
                           orientation="h", title="Your Model's Feature Importance")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Insight about top feature
                top_feature = importance_df.iloc[-1]
                st.markdown(f"""
                <div class="insight-card">
                    <h4>🏆 Your Model's Favorite Signal</h4>
                    <p>Your {selected_classifier} model found <strong>{top_feature['Feature']}</strong> to be the most important factor, 
                    using it for {top_feature['Importance']:.0%} of its decision-making process.</p>
                </div>
                """, unsafe_allow_html=True)

            # Actionable insights
            st.markdown("### 💡 Insights and Next Steps")

            insights = []

            if custom_accuracy > 0.8:
                insights.append("🌟 Excellent performance! Your model could be used for real predictions.")
            elif custom_accuracy > 0.7:
                insights.append("👍 Good performance! Consider fine-tuning parameters for even better results.")
            else:
                insights.append("🔧 Room for improvement. Try different time periods or algorithms.")

            if len(selected_phases) < 3:
                insights.append("🎭 You focused on specific phases - try including all phases to see the full picture.")

            if (years[1] - years[0]) < 10:
                insights.append("📅 Consider using more years of data for better model training.")

            for insight in insights:
                st.markdown(f"""
                <div class="insight-card">
                    <p>{insight}</p>
                </div>
                """, unsafe_allow_html=True)

            # Download results
            st.markdown("### 📥 Take Your Results With You")

            # Create results dataframe
            results_df = pd.DataFrame({
                'Date': X_test.index,
                'Actual_Phase': y_test,
                'Predicted_Phase': y_pred_custom,
                'Correct': y_test == y_pred_custom
            })

            results_df = results_df.merge(filtered_df[['Date'] + feature_cols],
                                        left_on='Date', right_index=True, how='left')

            csv_data = results_df.to_csv(index=False)

            st.download_button(
                "📥 Download Your Model's Predictions",
                data=csv_data,
                file_name=f"custom_enso_model_{selected_classifier.lower().replace(' ', '_')}_{years[0]}_{years[1]}.csv",
                mime="text/csv",
                help="Download detailed results including predictions and input features"
            )

# Footer with credits and additional info
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
           border-radius: 10px; color: white; margin-top: 2rem;">
    <h3>🌊 ENSOcast</h3>
    <p>Crafted with ❤️ by Dylan Dsouza</p>
    <p><em>Bringing climate science to life through storytelling and AI</em></p>
    <small>Data sources: NOAA, ECMWF | Built with Streamlit, Plotly, and Scikit-learn</small>
</div>
""", unsafe_allow_html=True)
