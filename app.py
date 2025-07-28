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

st.set_page_config(
    page_title="ENSOcast",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        font-size: 18px;
        text-align: center;
        color: black;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .story-card p {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    
    .description {
        text-align: left;
        padding-left: 2rem;
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
        text-align: center;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
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

def create_story_intro():
    st.markdown("""
    <div class="main-header">
        <h1>🌊 ENSOcast</h1>
        <h3>Decoding El Niño–Southern Oscillation</h3>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_model_and_data():
    try:
        df = pd.read_csv("merged_enso.csv", parse_dates=["Date"])
        model = joblib.load("ENSOcast_model.pkl")
        return df, model
    except FileNotFoundError:
        st.error("Data files not found. Please ensure merged_enso.csv and ENSOcast_model.pkl are in the correct directory.")
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

def create_feature_for_date(target_date, df, feature_cols):
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

data_loaded = True
try:
    df, model = load_model_and_data()
    if df is None or model is None:
        data_loaded = False
    else:
        sst_ds = load_sst_dataset()
        df = df.sort_values("Date").reset_index(drop=True)
        split_idx = int(0.8 * len(df))

        feature_cols = [
            "SST_Anomaly", "SOI", "SOI_lag_1", "SOI_lag_2", "SOI_lag_3",
            "SST_Anomaly_lag_1", "SST_Anomaly_lag_2", "SST_Anomaly_lag_3",
            "month_sin", "month_cos"
        ]
        X = df[feature_cols]
        y = df["ENSO_Label"]

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        total_predictions = len(y_test)
        correct_predictions = int(accuracy * total_predictions)

        label_map = {0: "La Niña", 1: "Neutral", 2: "El Niño"}

        df.loc[df.index[split_idx:], "Predicted_Phase"] = [label_map[i] for i in y_pred]

        df.loc[df.index[split_idx:], "True_Phase"] = [label_map[i] for i in y_test]
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

create_story_intro()

if not data_loaded:
    st.error("⚠️Unable to load required data files. Please check your setup.")
    st.stop()

st.sidebar.title("🌊 ENSOcast")
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Tab Navigation")
page = st.sidebar.radio(
    "",
    ["🌍 Understanding ENSO",
     "🌡️ Ocean Temperatures",
     "📊 Past Patterns",
        "🔬 Model Performance",
        "🛠️ Train Model",
        "🔮 Predict the Future"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("*Created by Dylan Dsouza*")

if page == "🌍 Understanding ENSO":

    st.markdown("""
    <div class="story-card">
        <h2>🌍 ENSO: The Pulse of the Pacific</h2>
        <p>Imagine the Pacific Ocean as the Earth's beating heart. Every few years, its rhythm shifts. Ocean temperatures rise or fall, winds change direction, and rainfall patterns grow unpredictable. 
        Storms intensify in some places and disappear in others. Crops either thrive or fail.</p> 
        <p>This natural cycle, called the <b>El Niño–Southern Oscillation (ENSO)</b>, is driven by fluctuations 
        in sea surface temperature and atmospheric pressure in the Pacific Ocean. It moves through three phases:</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        color = "#f6416c"
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
            <h3>🔴 El Niño</h3>
            <h4><em><strong>"The Little Boy"</strong></em></h4>
            <p>Ocean warms up, trade winds weaken, and rainfall increases in the eastern Pacific. This often brings destructive floods to California and droughts from Australia to South Asia.</p>
            <small>Every 2–7 years</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        color = "#94a3b8"
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
            <h3>⚪ Neutral</h3>
            <h4><em><strong>"The Quiet Phase"</strong></em></h4>
            <p>Ocean temperatures stay near average, and weather patterns remain stable—neither extreme nor unusual. This brings typical seasonal weather worldwide.</p>
            <small>Most frequent state</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        color = "#3b82f6"
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
            <h3>🔵 La Niña</h3>
            <h4><em><strong>"The Little Girl"</strong></em></h4>
            <p>Ocean cools down, and trade winds strengthen. This typically causes droughts in South America and fuel intense hurricanes in the Atlantic.</p>
            <small>Often follows El Niño</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="story-card">
        <h2>💥 The Global Impact of ENSO</h2>
        <p>ENSO may begin in the Pacific, but its ripple effects reach around the world:</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="background: #ff6b6b; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>🌾 Agriculture</h4>
                <p>Variations in rainfall patterns significantly affect crop production, food supply stability, and agricultural economies.</p>
            </div>
            <div style="background: #4ecdc4; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>🌤️ Weather</h4>
                <p>ENSO alters the distribution of extreme weather events, resulting in increased storms in some regions and decreased rainfall in others.</p>
            </div>
            <div style="background: #45b7d1; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>💰 Economy</h4>
                <p>Fluctuations in crop yields and energy demand influence financial markets worldwide and sway international trade.</p>
            </div>
            <div style="background: #f9ca24; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>🐠 Marine Life</h4>
                <p>Ocean temperature changes drive shifts in fish populations, degrade coral reef ecosystems, and disrupt marine food webs.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "🌡️ Ocean Temperatures":
    st.markdown("""
    <div class="story-card">
        <h2>🌡️ Ocean Temperatures at a Glance</h2>
        <p>Global sea surface temperatures tell us a story. In the heart of the Pacific, <b>Niño 3.4</b> reveals the pulse of ENSO. Defined by the coordinates <em>5°N-5°S</em> latitude and <em>170°W-120°W</em> longitude, this rectangular region is where subtle changes in warmth tell us if El Niño or La Niña is forming.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.slider("Select Year", min_value=1982, max_value=2024, value=2010, help="Drag to select the year you want to explore")
    with col2:
        month_dict = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        selected_month = st.selectbox("Select Month",
                                    list(month_dict.keys()), index=7, help="Select the month you want to explore")

    month_num = month_dict[selected_month]

    if sst_ds is not None:
        try:
            with st.spinner(f"Downloading data for {selected_month} {selected_year}..."):
                sst_slice = sst_ds.sel(time=(sst_ds['time.year'] == selected_year) &
                                     (sst_ds['time.month'] == month_num))['sst']

                fig, ax = plt.subplots(figsize=(12, 6))

                # Enhanced colormap and styling
                im = sst_slice.plot(ax=ax, cmap='coolwarm',
                                  cbar_kwargs={"label": "Sea Surface Temperature (°C)", "shrink": 1})
                ax.add_patch(patches.Rectangle((190, -5), 50, 10, edgecolor='black', facecolor='none', linewidth=1, linestyle='--'))
                ax.text(189, 8, 'Niño 3.4 Region', color='black')

                ax.set_title(f'Sea Surface Temperature - {selected_month} {selected_year}',
                           fontsize=13, pad=10)
                ax.set_xlabel("Longitude (°E)", fontsize=12)
                ax.set_ylabel("Latitude (°N)", fontsize=12)

                max_temp = float(sst_slice.max())
                avg_temp = float(sst_slice.mean())
                min_temp = float(sst_slice.min())

                st.pyplot(fig)

                col1, col2, col3 = st.columns(3)
                with col1:
                    color = "#f6416c"
                    st.markdown(f"""
                    <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40);">
                        <h3>🔥 Hottest</h3>
                        <h2>{max_temp:.1f}°C</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    color = "#94a3b8"
                    st.markdown(f"""
                    <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40);">
                        <h3>🌡️ Average</h3>
                        <h2>{avg_temp:.1f}°C</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    color = "#3b82f6"
                    st.markdown(f"""
                    <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40);">
                        <h3>🧊 Coolest</h3>
                        <h2>{min_temp:.1f}°C</h2>
                    </div>
                    """, unsafe_allow_html=True)

                date_context = df[df['Date'].dt.year == selected_year]
                if len(date_context) > 0:
                    monthly_data = date_context[date_context['Date'].dt.month == month_num]
                    if len(monthly_data) > 0:
                        phase = monthly_data.iloc[0]['ENSO_Phase']
                        oni_value = monthly_data.iloc[0]['ONI']

                        phase_colors = {"El Niño": "#f6416c", "La Niña": "#3b82f6", "Neutral": "#94a3b8"}
                        phase_emojis = {"El Niño": "🔴", "La Niña": "🔵", "Neutral": "⚪"}

                        article = "an" if phase.startswith("El") else "a"

                        st.markdown(f"""
                            <div class="insight-card" style="background: {phase_colors[phase]}20;">
                                <h3>{phase_emojis[phase]} {selected_month} {selected_year} was {article} <strong>{phase}</strong> month with an ONI value of <strong>{oni_value:.2f}</strong></h3>
                            </div>
                            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"🌊 Unable to load ocean data for {selected_month} {selected_year}: {e}")
            st.info("💡 Try selecting a different date, or check your internet connection for satellite data.")
    else:
        st.warning("🛰️ Live satellite data unavailable.")

elif page == "📊 Past Patterns":
    st.markdown("""
    <div class="story-card">
        <h2>📊 The Data Behind ENSO</h2>
        <p>To understand the future of ENSO, we first need to examine its past. These charts show how ENSO phases have shifted over time, highlighting patterns, trends, and seasonal tendencies.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        years = st.slider(
            "Select Year Range", 1982, 2024, (2000, 2020),
            help="Drag to select the time period you want to explore"
        )

    with col2:
        selected_phases = st.multiselect(
            "Select ENSO Phases",
            ["El Niño", "Neutral", "La Niña"],
            default=["El Niño", "Neutral", "La Niña"],
            help="Select which ENSO phases to include in your analysis"
        )

    df_filtered = df[
        (df["Date"].dt.year >= years[0]) &
        (df["Date"].dt.year <= years[1]) &
        (df["ENSO_Phase"].isin(selected_phases))
    ]

    if len(df_filtered) == 0:
        st.warning("No data available for your selected criteria. Try adjusting your filters.")
        st.stop()

    phase_counts = df_filtered["ENSO_Phase"].value_counts()
    total_months = len(df_filtered)
    years_span = years[1] - years[0] + 1

    if years[0] == years[1]:
        st.markdown(f"""
            <div class="story-card">
                <h2>🕰️ Your Selected Timeline:</h2>
                <h3>January {years[0]} – December {years[1]}</h3>
                <p>In this <strong>1 year</strong>, here's how ENSO spent its time:</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="story-card">
                <h2>🕰️ Your Selected Timeline:</h2>
                <h3>January {years[0]} – December {years[1]}</h3>
                <p>In these <strong>{years_span} years</strong>, here's how ENSO spent its time:</p>
            </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for i, (phase, count) in enumerate(phase_counts.items()):
        percentage = (count / total_months) * 100
        color = "#f6416c" if phase == "El Niño" else "#3b82f6" if phase == "La Niña" else "#94a3b8"
        emoji = "🔴" if phase == "El Niño" else "🔵" if phase == "La Niña" else "⚪"

        years = count // 12
        months = count % 12

        if years > 0 and months > 0:
            time_str = f"{years} year{'s' if years > 1 else ''} {months} month{'s' if months > 1 else ''}"
        elif years > 0:
            time_str = f"{years} year{'s' if years > 1 else ''}"
        else:
            time_str = f"{months} month{'s' if months > 1 else ''}"

        if i == 0:
            col = col2
        elif i == 1:
            col = col3
        else:
            col = col1

        with col:
            st.markdown(f"""
            <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40);">
                <h3>{emoji} {phase}</h2>
                <h4>{time_str}</h4>
                <p>{percentage:.1f}% of the time</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🌡️ Sea Surface Temperature (SST): The First ENSO Signal")

    st.markdown("""
    Sea Surface Temperature (SST) refers to the temperature of water measured at the ocean's surface. In the Niño 3.4 region, deviations from the climatological mean (the expected long-term seasonal average) provide early indications of ENSO development.""")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["SST"],
            name="Ocean Temperature",
            line=dict(color='skyblue', width=2),
            hovertemplate="<b>%{x}</b><br>Temperature: %{y:.2f}°C<extra></extra>"
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["SST_Climatology"],
            name="Expected Temperature",
            line=dict(color='orange', dash='dot', width=2),
            hovertemplate="<b>%{x}</b><br>Expected: %{y:.2f}°C<extra></extra>"
        ),
        secondary_y=False,
    )

    fig.update_layout(
        xaxis_title="Date",
        template="plotly_white",
        height=400,
        margin=dict(t=30, b=30),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.add_hrect(y0=27.75, y1=29.57, line_width=0, fillcolor="red", opacity=0.1)
    fig.add_hrect(y0=24.69, y1=26.45, line_width=0, fillcolor="blue", opacity=0.1)
    fig.add_hline(y=27.75, line_dash="dot", line_color="red", annotation_text="El Niño Conditions", annotation_position="top right")
    fig.add_hline(y=26.45, line_dash="dot", line_color="blue", annotation_text="La Niña Conditions", annotation_position="bottom right")
    fig.update_yaxes(title_text="Sea Surface Temperature (°C)", secondary_y=False)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-card" style="text-align: center;">
        <h4>🌡️ <strong>Warm or cool waters?</strong></h4>
        <p>SST anomalies above the climatological mean indicate the onset of El Niño, while those below suggest La Niña.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🌀 Southern Oscillation Index (SOI): Pressure Patterns Driving ENSO")

    st.markdown("""
    The Southern Oscillation Index (SOI) is a standardized scale which tracks sea level pressure differences across the Pacific, specifically between Tahiti and Darwin, Australia. These pressure differences steer wind patterns and can trigger or suppress ENSO events.
    """)

    fig_soi = px.line(df_filtered, x="Date", y="SOI")
    fig_soi.update_layout(
        yaxis_title="Southern Oscillation Index",
        template="plotly_dark",
        height=400,
        margin=dict(t=30, b=30)
    )
    fig_soi.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="La Niña Conditions", annotation_position="top right")
    fig_soi.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="El Niño Conditions", annotation_position="bottom right")
    fig_soi.add_hrect(y0=0, y1=3.2, line_width=0, fillcolor="blue", opacity=0.1)
    fig_soi.add_hrect(y0=-3.2, y1=0, line_width=0, fillcolor="red", opacity=0.1)

    fig_soi.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["SOI"],
            name="SOI",
            showlegend=False,
            line=dict(color='skyblue', width=2),
            hovertemplate="<b>%{x}</b><br>SOI: %{y:.2f}<extra></extra>"
        ),
        secondary_y=False,
    )

    st.plotly_chart(fig_soi, use_container_width=True)

    st.markdown("""
    <div class="insight-card" style="text-align: center;">
        <h4>🌀 <strong>High or low pressure?</strong></h4>
        <p>A deeply negative SOI value generally supports El Niño conditions, while strong positive values align with the onset of La Niña.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📈 Oceanic Niño Index (ONI): Quantifying ENSO")

    st.markdown("""
    The Oceanic Niño Index (ONI) tracks the three-month moving average of SST anomalies in the Niño 3.4 region. Widely regarded as the gold standard, it serves as the primary benchmark for identifying and classifying ENSO events.
    """)
    fig_oni = go.Figure()

    fig_oni.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["ONI"],
            name="ONI",
            line=dict(color='skyblue', width=2),
            hovertemplate="<b>%{x|%b %Y}</b><br>ONI: %{y:.2f}<extra></extra>"
        )
    )

    fig_oni.update_layout(
        xaxis_title="Date",
        yaxis_title="Oceanic Niño Index",
        height=400,
        yaxis_range=[-2.7, 2.7],
        template="plotly_dark",
        margin=dict(t=30, b=30)
    )

    fig_oni.add_hline(y=0.5, line_dash="dot", line_color="red",
                      annotation_text="El Niño Threshold", annotation_position="top right")
    fig_oni.add_hline(y=-0.5, line_dash="dot", line_color="blue",
                      annotation_text="La Niña Threshold", annotation_position="bottom right")

    fig_oni.add_hrect(y0=0.5, y1=2.7, line_width=0, fillcolor="red", opacity=0.1)
    fig_oni.add_hrect(y0=-2.7, y1=-0.5, line_width=0, fillcolor="blue", opacity=0.1)

    st.plotly_chart(fig_oni, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <h4>📈 <strong>Positive or negative anomaly?</strong></h4> 
        <p>If ONI exceeds +0.5 for months, El Niño is underway; if it remains below -0.5, La Niña is active. Values between these thresholds indicate neutral conditions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    phase_numeric = df_filtered["ENSO_Phase"].map({"El Niño": 1, "Neutral": 0, "La Niña": -1})

    fig_phases = go.Figure()

    colors = {"La Niña": "#3b82f6", "Neutral": "gray", "El Niño": "#ef4444"}
    st.markdown("### 🗓️ Seasonal Behavior of ENSO")

    st.markdown("""
    Some months are more likely to host El Niño or La Niña events. This bar chart reveals the seasonal rhythm of each ENSO phase.
    """)
    df_filtered['Month'] = df_filtered['Date'].dt.month
    seasonal_patterns = df_filtered.groupby(['Month', 'ENSO_Phase']).size().unstack(fill_value=0)

    fig_seasonal = px.bar(
        seasonal_patterns.reset_index().melt(id_vars='Month', var_name='Phase', value_name='Count'),
        x='Month', y='Count', color='Phase',
        color_discrete_map=colors
    )

    fig_seasonal.update_layout(
        margin=dict(t=30, b=30),
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
        <h4>🗓️ <strong>Which seasons matter most?</strong></h4> 
        <p>ENSO seems to show asymmetric seasonal patterns — El Niño events cluster in late fall/winter months, while La Niña strengthens in winter and tends to persist longer. Neutral conditions are more common during spring/summer transition periods when phase changes typically occur.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "🔬 Model Performance":
    st.markdown("""
    <div class="story-card">
       <h2>🔬 Understanding the Baseline Model</h2>
       <p>ENSOcast is a climate prediction tool that leverages a <strong>Random Forest Classifier</strong> to forecast monthly ENSO phases based on patterns in ocean temperatures, atmospheric pressure, and seasonal cycles.</p>
    
    <p class="description"><strong>Model Targets —</strong> ENSO phase classifications derived from the <strong>Oceanic Niño Index (ONI)</strong>:</p>
    
    <ul class="description" style="text-align: left;">
        <li class="description"><strong>El Niño</strong></li>
        <li class="description"><strong>La Niña</strong></li>
        <li class="description"><strong>Neutral</strong></li>
    </ul>
    
    <p class="description"><strong>Key Input Features —</strong> Critical climate indicators that capture ocean-atmosphere dynamics:</p>
    
    <ul class="description" style="text-align: left;">
       <li><strong>Sea Surface Temperatures (SST)</strong> in the Niño 3.4 region, the primary oceanic signal of ENSO variability</li>
       <li class="description"><strong>Southern Oscillation Index (SOI)</strong>, measuring atmospheric pressure differences across the Pacific</li>
    </ul>
    
    <p class="description"><strong>Advanced Feature Engineering —</strong> To capture temporal patterns and seasonal dependencies, ENSOcast applies several preprocessing techniques:</p>
    
    <ul class="description" style="text-align: left;">
       <li>Calculating <strong>SST anomalies</strong> by eliminating the long-term climatological baseline</li>
       <li>Creating <strong>lagged variables</strong> (1–3 months) for both SOI and SST anomalies to capture predictive relationships</li>
       <li>Encoding <strong>seasonal cycles</strong> using sine and cosine transformations of calendar months</li>
    </ul>
    
    <p class="description"><strong>Validation Approach —</strong> ENSOcast is trained on the earliest 80% of historical ENSO data and achieves <strong>83% accuracy</strong> when predicting the most recent 20% of observations.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        color = "#94a3b8"
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
            <h3>🎯 Overall Accuracy</h3>
            <h2>{accuracy:.0%}</h2>
            <p>Correct predictions</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
            <h3>📊 Total Tested</h3>
            <h2>{total_predictions:,}</h2>
            <p>Months analyzed</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
            <h3>✅ Success Stories</h3>
            <h2>{correct_predictions:,}</h2>
            <p>Months predicted correctly</p>
        </div>
        """, unsafe_allow_html=True)

    # Classification report in a more narrative format
    report = classification_report(df["True_Phase"], df["Predicted_Phase"], output_dict=True)

    phases = ["El Niño", "Neutral", "La Niña"]
    phase_colors = {"El Niño": "#f6416c", "Neutral": "#94a3b8", "La Niña": "#3b82f6"}
    phase_emojis = {"El Niño": "🔴", "Neutral": "⚪", "La Niña": "🔵"}

    for phase in phases:
        if phase in report:
            precision = report[phase]['precision']
            recall = report[phase]['recall']
            f1 = report[phase]['f1-score']
            support = int(report[phase]['support'])

            st.markdown(f"""
            <div class="insight-card" style="background: linear-gradient(135deg, {phase_colors[phase]}20, {phase_colors[phase]}40);">
                <h4>{phase_emojis[phase]} {phase} Performance</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                    <div>
                        <strong>Precision:</strong> {precision:.0%}<br>
                        <small>When the model predicts "{phase}", it is correct {precision:.0%} of the time.</small>
                    </div>
                    <div>
                        <strong>Recall:</strong> {recall:.0%}<br>
                        <small>The model correctly identifies {recall:.0%} of actual {phase} months.</small>
                    </div>
                    <div>
                        <strong>F1-Score:</strong> {f1:.0%}<br>
                        <small>This is a balanced metric combining precision and recall.</small>
                    </div>
                    <div>
                        <strong>Sample Size:</strong> {support}<br>
                        <small>In the dataset, {support} months were labeled as {phase}.</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Confusion matrix as a story
    st.markdown("### 🧮 Confusion Matrix")

    cm = confusion_matrix(df["True_Phase"], df["Predicted_Phase"], labels=["El Niño", "Neutral", "La Niña"])

    # Create a heatmap-style visualization
    fig_cm = px.imshow(cm,
                       labels=dict(x="Predicted Phase", y="Actual Phase", color="Count"),
                       x=["🔴 El Niño", "⚪ Neutral", "🔵 La Niña"],
                       y=["🔴 El Niño", "⚪ Neutral", "🔵 La Niña"],
                       color_continuous_scale="Reds")

    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            fig_cm.add_annotation(x=j, y=i, text=str(cm[i][j]),
                                showarrow=False, font_size=16, font_color="white" if cm[i][j] > cm.max()/2 else "black")

    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <p>🧮 <strong>Reading the confusion:</strong> The diagonal shows correct predictions (darker = better). 
        Off-diagonal squares show mistakes - when the model confused one phase for another.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

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

elif page == "🛠️ Train Model":
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
                    <div class="insight-card" style="background: linear-gradient(135deg, {color}20, {color}40);>
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

elif page == "🔮 Predict the Future":
    st.header("🔮 Advanced ENSO Predictions")

    # Introduction section
    st.markdown("""
    ### How This Prediction Works
    Our AI model analyzes sea surface temperatures, atmospheric pressure, and historical patterns to predict which ENSO phase is most likely for any given month.
    """)

    st.markdown("---")

    # Main prediction interface
    st.markdown("### 🎯 Predict ENSO Phase for Any Month")
    st.markdown("Choose a month and year to see what ENSO phase is predicted, along with the model's confidence level.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        target_year = st.number_input("Year", min_value=1982, max_value=2030, value=2024, help="Select any year from 1982 to 2030")
    with col2:
        target_month = st.selectbox("Month", [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ], index=0, help="Select the month you want to predict")
    with col3:
        st.markdown("&nbsp;")  # Spacer
        predict_button = st.button("🔮 Make Prediction", type="primary")

    if predict_button:
        month_num = {
            "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
            "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
        }[target_month]

        target_date = datetime.date(target_year, month_num, 1)

        try:
            # Create features for the target date
            X_target = create_feature_for_date(target_date, df, feature_cols)

            # Get prediction and probabilities
            prediction = model.predict(X_target)[0]
            probabilities = model.predict_proba(X_target)[0]

            predicted_phase = label_map[prediction]
            max_prob = max(probabilities)

            # Create result display
            if predicted_phase == "El Niño":
                phase_emoji = "🔴"
                phase_color = "red"
                phase_description = "Warmer ocean temperatures expected. This often brings increased rainfall to the southern US and can disrupt normal weather patterns globally."
            elif predicted_phase == "La Niña":
                phase_emoji = "🔵"
                phase_color = "blue"
                phase_description = "Cooler ocean temperatures expected. This often brings drier conditions to the southern US and more active hurricane seasons."
            else:
                phase_emoji = "⚪"
                phase_color = "gray"
                phase_description = "Normal ocean temperatures expected. Weather patterns should be closer to typical seasonal averages."

            st.markdown("### 📊 Prediction Results")

            # Main prediction result
            st.markdown(f"""
            <div style="background-color: {phase_color}15; padding: 20px; border-radius: 10px; border-left: 5px solid {phase_color};">
                <h2 style="color: {phase_color}; margin: 0;">{phase_emoji} {predicted_phase}</h2>
                <p style="margin: 5px 0 0 0; font-size: 18px;"><strong>Predicted for {target_month} {target_year}</strong></p>
                <p style="margin: 10px 0 0 0;">{phase_description}</p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence level
            if max_prob > 0.7:
                confidence = "High"
                confidence_color = "green"
                confidence_desc = "The model is very confident in this prediction."
            elif max_prob > 0.5:
                confidence = "Moderate"
                confidence_color = "orange"
                confidence_desc = "The model has reasonable confidence, but there's some uncertainty."
            else:
                confidence = "Low"
                confidence_color = "red"
                confidence_desc = "The model has low confidence - the prediction is uncertain."

            st.markdown(f"""
            **Model Confidence:** <span style="color: {confidence_color};">**{confidence}** ({max_prob:.1%})</span>  
            *{confidence_desc}*
            """, unsafe_allow_html=True)

            # Detailed probabilities
            st.markdown("### 📈 Detailed Probabilities")
            st.markdown("Here's how confident the model is for each possible ENSO phase:")

            prob_data = {
                "🔵 La Niña": probabilities[0],
                "⚪ Neutral": probabilities[1],
                "🔴 El Niño": probabilities[2]
            }

            for phase, prob in prob_data.items():
                st.progress(prob, text=f"{phase}: {prob:.1%}")

            # What this means section
            st.markdown("### 🤔 What Does This Mean?")
            st.markdown(f"""
            - **Most Likely Outcome**: {predicted_phase} conditions in {target_month} {target_year}
            - **Confidence Level**: {confidence} - {confidence_desc.lower()}
            - **Key Insight**: The model analyzed sea surface temperatures, atmospheric pressure patterns, and historical data to make this prediction
            """)

            if max_prob < 0.6:
                st.warning("⚠️ **Note**: This prediction has moderate to low confidence. ENSO predictions become less reliable further into the future or during transition periods.")

        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
            st.info("Please try a different date or check if the data is available for your selected time period.")

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

    st.markdown("### 🎯 Predict or Cast Your ENSO Phase")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        target_year = st.number_input(
            "🗓️ Select Year", min_value=1982, max_value=2030, value=2024,
            help="Choose any year from 1982 to 2030"
        )
    with col2:
        target_month = st.selectbox(
            "🌙 Select Month", [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ],
            index=datetime.datetime.now().month - 1,
            help="Choose the month you want to predict"
        )
    with col3:
        st.markdown("&nbsp;")  # spacer
        predict_button = st.button("🔮 Reveal the Future", type="primary", use_container_width=True)

    if predict_button:
        month_num = {
            "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
            "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
        }[target_month]

        target_date = datetime.date(target_year, month_num, 1)

        with st.spinner("🌊 Consulting the ocean spirits... Reading the signs..."):
            try:
                X_target = create_feature_for_date(target_date, df, feature_cols)
                prediction = model.predict(X_target)[0]
                probabilities = model.predict_proba(X_target)[0]

                predicted_phase = label_map[prediction]
                max_prob = max(probabilities)

                # Dramatic reveal
                st.balloons()

                # Phase display with gradients and descriptions
                phase_styles = {
                    "El Niño": {
                        "emoji": "🔴",
                        "bg": "linear-gradient(135deg, #ff9a8b 0%, #f6416c 100%)",
                        "description": (
                            "The ocean will run warm with El Niño's fire. "
                            "Expect the unexpected — flooding rains in some lands, drought in others."
                        )
                    },
                    "La Niña": {
                        "emoji": "🔵",
                        "bg": "linear-gradient(135deg, #a8edea 0%, #3b82f6 100%)",
                        "description": (
                            "The ocean will run cold under La Niña's influence. "
                            "Hurricanes may dance with greater fury, and weather patterns will intensify."
                        )
                    },
                    "Neutral": {
                        "emoji": "⚪",
                        "bg": "linear-gradient(135deg, #d299c2 0%, #fef9d3 100%)",
                        "description": (
                            "The ocean rests in balance. Weather patterns will follow "
                            "their seasonal rhythms without dramatic shifts."
                        )
                    }
                }

                style = phase_styles.get(predicted_phase, phase_styles["Neutral"])

                st.markdown(f"""
                <div style="
                    background: {style['bg']};
                    color: #333;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                ">
                    <h1>{style['emoji']} {predicted_phase} Awakens</h1>
                    <h3>For {target_month} {target_year}</h3>
                    <p style="font-size: 1.2em;">{style['description']}</p>
                    <p><strong>Oracle's Confidence:</strong> {max_prob:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

                # Plotly bar chart for detailed probabilities
                prob_data = pd.DataFrame({
                    'Phase': ['🔵 La Niña', '⚪ Neutral', '🔴 El Niño'],
                    'Probability': probabilities,
                    'Colors': ['#3b82f6', '#94a3b8', '#f6416c']
                })

                fig = px.bar(
                    prob_data, x='Phase', y='Probability',
                    color='Colors', color_discrete_map='identity',
                    title="The Oracle's Vision - Detailed Probabilities"
                )
                fig.update_layout(showlegend=False, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                # Confidence poetic interpretation
                if max_prob > 0.8:
                    confidence_story = "🎯 **Crystal Clear Vision** - The signs are unmistakable"
                elif max_prob > 0.6:
                    confidence_story = "👁️ **Strong Intuition** - The patterns point clearly in one direction"
                elif max_prob > 0.4:
                    confidence_story = "🤔 **Clouded Vision** - The future remains uncertain, multiple paths possible"
                else:
                    confidence_story = "🌫️ **Misty Prophecy** - The ocean spirits are conflicted"

                st.markdown(f"""
                <div style="
                    background: #f0f4f8;
                    border-left: 5px solid #007acc;
                    padding: 15px;
                    border-radius: 5px;
                    font-style: italic;
                    margin-bottom: 20px;
                ">
                    {confidence_story}
                    <br><br>
                    <em>"The further we peer into time's river, the murkier the waters become. 
                    Use this wisdom as a guide, not gospel."</em>
                </div>
                """, unsafe_allow_html=True)

                # What this means / explanation section
                st.markdown("### 🤔 What Does This Mean?")
                st.markdown(f"""
                - **Most Likely Outcome**: {predicted_phase} conditions in {target_month} {target_year}  
                - **Confidence Level**: {confidence_story.split(' - ')[0].strip('🎯👁️🤔🌫️**')}  
                - **Key Insight**: The model analyzed sea surface temperatures, atmospheric pressure patterns, and historical data to make this prediction.
                """)

                if max_prob < 0.6:
                    st.warning(
                        "⚠️ **Note**: This prediction has moderate to low confidence. ENSO predictions become less reliable further into the future or during transition periods."
                    )

            except Exception as e:
                st.error(f"❌ The oracle's vision is clouded: {e}")
                st.info("Please try a different date or check if the data is available for your selected time period.")


    st.markdown("---")

    # Educational section about model performance
    st.markdown("### 📚 How Accurate Are These Predictions?")

    # Calculate and display model accuracy in an intuitive way
    accuracy = accuracy_score(df["True_Phase"], df["Predicted_Phase"])
    total_predictions = len(df)
    correct_predictions = int(accuracy * total_predictions)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Accuracy", f"{accuracy:.1%}", help="Percentage of predictions that were correct")
    with col2:
        st.metric("Total Predictions", f"{total_predictions:,}", help="Number of months analyzed")
    with col3:
        st.metric("Correct Predictions", f"{correct_predictions:,}", help="Number of months predicted correctly")

    st.markdown(f"""
    **What this means**: Out of {total_predictions:,} months of historical data, our model correctly predicted the ENSO phase {correct_predictions:,} times. 
    That's an accuracy rate of {accuracy:.1%}, which is quite good for climate prediction!
    
    **Important Notes**:
    - Climate prediction is inherently uncertain - even the best models can't be 100% accurate
    - Predictions are more reliable for the near future (3-6 months) than long-term forecasts
    - The model works best during stable climate periods and may be less accurate during rapid transitions
    """)

    # Simple visualization of recent predictions vs reality
    st.markdown("### 📊 Recent Predictions vs Reality")
    st.markdown("See how well the model has been performing recently:")

    # Show last 2 years of data
    recent_data = df[df["Date"] >= (df["Date"].max() - pd.DateOffset(years=2))].copy()

    # Create a simple comparison chart
    fig_recent = go.Figure()

    # Add actual phases
    fig_recent.add_trace(go.Scatter(
        x=recent_data["Date"],
        y=[1 if phase == "El Niño" else 0 if phase == "Neutral" else -1 for phase in recent_data["True_Phase"]],
        mode='lines+markers',
        name='Actual Phase',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))

    # Add predicted phases
    fig_recent.add_trace(go.Scatter(
        x=recent_data["Date"],
        y=[1 if phase == "El Niño" else 0 if phase == "Neutral" else -1 for phase in recent_data["Predicted_Phase"]],
        mode='lines+markers',
        name='Predicted Phase',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=6, symbol='x')
    ))

    fig_recent.update_layout(
        title="Model Predictions vs Actual ENSO Phases (Last 2 Years)",
        xaxis_title="Date",
        yaxis_title="ENSO Phase",
        yaxis=dict(
            tickmode='array',
            tickvals=[-1, 0, 1],
            ticktext=['La Niña 🔵', 'Neutral ⚪', 'El Niño 🔴']
        ),
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig_recent, use_container_width=True)

    st.markdown("""
    **How to read this chart**:
    - **Blue line**: What actually happened
    - **Red dotted line**: What the model predicted
    - When the lines overlap, the model was correct
    - When they diverge, the model made an error
    """)

    # Final tips section
    st.markdown("### 💡 Tips for Using These Predictions")
    st.markdown("""
    1. **Short-term predictions** (1-3 months ahead) are generally more reliable
    2. **High confidence predictions** (>70%) are more likely to be accurate
    3. **Consider multiple factors** - ENSO is just one part of the climate system
    4. **Use for planning** - These predictions can help with agricultural, business, or travel planning
    5. **Stay updated** - Climate patterns can change rapidly, so check back regularly
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
           border-radius: 10px; color: white; margin-top: 2rem;">
    <h3>🌊 ENSOcast</h3>
    <p>Decoding El Niño–Southern Oscillation</p>
    <p><em>Created by Dylan Dsouza</em></p>
    <small>Data source: NOAA | Built with Streamlit, plotly, and scikit-learn</small>
</div>
""", unsafe_allow_html=True)
