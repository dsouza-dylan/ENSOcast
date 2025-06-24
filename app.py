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
    st.warning("SHAP not available. Install with: pip install shap")

st.set_page_config(page_title="ENSOcast", layout="wide")

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

@st.cache_data
def calculate_cross_validation_scores(X, y):
    """Calculate cross-validation scores for the model"""
    model_cv = RandomForestClassifier(random_state=42, n_estimators=100)
    cv_scores = cross_val_score(model_cv, X, y, cv=5, scoring='accuracy')
    return cv_scores.mean(), cv_scores.std()

@st.cache_data
def prepare_shap_values(model, X_sample):
    """Prepare SHAP values for feature explanation"""
    if not SHAP_AVAILABLE:
        return None, None

    # Use a sample for SHAP to avoid performance issues
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values

def create_feature_for_date(target_date, df, feature_cols):
    """Create features for a specific date based on historical patterns"""
    # Convert target_date to datetime if it's not already
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)

    # Find the closest available data point
    df_sorted = df.sort_values('Date')
    closest_idx = np.argmin(np.abs((df_sorted['Date'] - target_date).dt.days))

    # Get seasonal components
    month = target_date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Use recent historical averages for other features
    recent_data = df_sorted.iloc[max(0, closest_idx-12):closest_idx+1]

    if len(recent_data) == 0:
        # Fallback to overall averages
        feature_values = df[feature_cols].mean()
    else:
        feature_values = recent_data[feature_cols].mean()

    # Update seasonal components
    feature_values['month_sin'] = month_sin
    feature_values['month_cos'] = month_cos

    return feature_values.values.reshape(1, -1)

# Load data
df, model = load_model_and_data()
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

st.title("🌊 ENSOcast: El Niño–Southern Oscillation Forecasts")

st.sidebar.title("🌊 ENSOcast")
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Tab Navigation")
page = st.sidebar.radio(
    "",
    ["🌡 Global SST Snapshot", "📈 Historical Trends", "🔮 Advanced Predictions", "🛠 Custom Model Training"],
    index=0
)
st.sidebar.markdown("### ")
st.sidebar.markdown("---")
st.sidebar.markdown("Made by Dylan Dsouza")

if page == "🌡 Global SST Snapshot":
    st.header("🌡 Global Sea Surface Temperature (SST) Snapshot")
    # st.markdown("### Global SST Snapshot")
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
        fig, ax = plt.subplots(figsize=(12, 6))
        sst_slice.plot(ax=ax, cmap='coolwarm', cbar_kwargs={"label": "°C"})
        ax.add_patch(patches.Rectangle((190, -5), 50, 10, edgecolor='black', facecolor='none', linewidth=1))
        ax.text(189, 8, 'Niño 3.4 Region', color='black')
        ax.set_xlabel("Longitude [°E]")
        ax.set_ylabel("Latitude [°N]")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to fetch SST data for {selected_month} {selected_year}. Error: {e}")

elif page == "📈 Historical Trends":
    st.header("📈 Historical Trends")

    years = st.slider("Select Year Range", 1982, 2025, (2000, 2020))
    selected_phases = st.multiselect(
        "Select ENSO Phases", ["La Niña", "Neutral", "El Niño"],
        default=["La Niña", "Neutral", "El Niño"]
    )

    df_filtered = df[
        (df["Date"].dt.year >= years[0]) &
        (df["Date"].dt.year <= years[1]) &
        (df["ENSO_Phase"].isin(selected_phases))
    ]

    st.markdown("### Sea Surface Temperature (SST) Timeline")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"], y=df_filtered["SST_Climatology"],
            name="Climatological SST (°C)",
            line=dict(color='deepskyblue', dash='dot')
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"], y=df_filtered["SST"],
            name="Observed SST (°C)",
            line=dict(color='orange')
        ),
        secondary_y=False,
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sea Surface Temperature (°C)",
        legend=dict(x=0.01, y=1.1),
        template="plotly_dark",
        margin=dict(t=30, b=30)
    )

    fig.update_yaxes(title_text="Sea Surface Temperature (°C)", range=[24, 30], secondary_y=False)
    fig.update_yaxes(range=[24, 30], secondary_y=True, showticklabels=False)

    climatology_min = df_filtered["SST_Climatology"].min()
    climatology_max = df_filtered["SST_Climatology"].max()

    fig.add_hline(y=climatology_min, line_dash="dot", line_color="gray",
                  annotation_text="Min. Climatological SST", annotation_position="bottom right")

    fig.add_hline(y=climatology_max, line_dash="dot", line_color="gray",
                  annotation_text="Max. Climatological SST", annotation_position="top right")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Oceanic Niño Index (ONI) Timeline")
    fig_oni = px.line(df_filtered, x="Date", y="ONI")
    fig_oni.update_yaxes(title_text="Oceanic Niño Index")
    fig_oni.add_hline(y=0.5, line_dash="dot", line_color="red",
                      annotation_text="El Niño Threshold", annotation_position="top right")
    fig_oni.add_hline(y=-0.5, line_dash="dot", line_color="blue",
                      annotation_text="La Niña Threshold", annotation_position="bottom right")

    st.plotly_chart(fig_oni, use_container_width=True)

    st.markdown("### Southern Oscillation Index (SOI) Timeline")
    fig_soi = px.line(df_filtered, x="Date", y="SOI")
    fig_soi.update_layout(
        yaxis_title="Southern Oscillation Index",
        template="plotly_dark",
        margin=dict(t=30, b=30)
    )
    fig_soi.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="La Niña Conditions", annotation_position="top right")
    fig_soi.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="El Niño Conditions", annotation_position="bottom right")
    fig_soi.add_hrect(y0=0, y1=3.2, line_width=0, fillcolor="blue", opacity=0.1)
    fig_soi.add_hrect(y0=-3.2, y1=0, line_width=0, fillcolor="red", opacity=0.1)
    st.plotly_chart(fig_soi, use_container_width=True)

elif page == "💡 Model Insights":
    st.header("💡 Model Insights")

    accuracy = accuracy_score(df["True_Phase"], df["Predicted_Phase"])

    # Original accuracy metric (unchanged)
    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    # NEW: Cross-validation scores (added, not replacing)
    with st.spinner("Calculating cross-validation scores..."):
        cv_mean, cv_std = calculate_cross_validation_scores(X, y_true)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("CV Mean Accuracy", f"{cv_mean * 100:.2f}%")
    with col2:
        st.metric("CV Std Dev", f"±{cv_std * 100:.2f}%")

    st.info(f"Cross-validation provides a more robust estimate: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")

    st.markdown("### Classification Report")
    report = classification_report(df["True_Phase"], df["Predicted_Phase"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(df["True_Phase"], df["Predicted_Phase"], labels=["La Niña", "Neutral", "El Niño"])
    st.dataframe(pd.DataFrame(cm, index=["True La Niña", "True Neutral", "True El Niño"], columns=["Pred La Niña", "Pred Neutral", "Pred El Niño"]))

    st.markdown("### Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    fig3 = px.bar(importance_df, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Download ENSO Predictions Results")
    st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="model_enso_predictions.csv", mime="text/csv")

elif page == "🛠 Custom Model Training":
    st.header("🛠 Custom Model Training")
    st.markdown("### Experiment with Different Models")
    st.info("Train and compare different machine learning models on filtered ENSO data to see which performs best for your specific time period and conditions.")

    years = st.slider("Select Year Range", 1982, 2025, (2000, 2020))
    selected_phases = st.multiselect("Select ENSO Phases", ["La Niña", "Neutral", "El Niño"], default=["La Niña", "Neutral", "El Niño"])

    filtered_df = df[
        (df["Date"].dt.year >= years[0]) &
        (df["Date"].dt.year <= years[1]) &
        (df["True_Phase"].isin(selected_phases))
    ]

    X_custom = filtered_df[feature_cols]
    y_custom = filtered_df["True_Phase"]

    X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.3, shuffle=False)

    # NEW: Classifier selection (default unchanged)
    classifier_options = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42, probability=True),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
    }

    selected_classifier = st.selectbox("Choose Classifier", list(classifier_options.keys()), index=0)
    custom_model = classifier_options[selected_classifier]

    # Brief explanation for each classifier
    classifier_info = {
        "Random Forest": "Ensemble method using multiple decision trees. Good for feature importance and handles non-linear relationships well.",
        "SVM": "Support Vector Machine finds optimal decision boundaries. Good for high-dimensional data.",
        "Logistic Regression": "Linear model for classification. Simple, interpretable, and fast."
    }
    st.info(classifier_info[selected_classifier])

    custom_model.fit(X_train, y_train)
    y_pred_custom = custom_model.predict(X_test)

    custom_accuracy = accuracy_score(y_test, y_pred_custom)
    st.metric("Custom Model Accuracy", f"{custom_accuracy * 100:.2f}%")

    st.markdown("### Classification Report")
    report = classification_report(y_test, y_pred_custom, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_custom, labels=["La Niña", "Neutral", "El Niño"])
    st.dataframe(pd.DataFrame(cm, index=["True La Niña", "True Neutral", "True El Niño"],
                              columns=["Pred La Niña", "Pred Neutral", "Pred El Niño"]))

    # Feature importance (when available)
    if hasattr(custom_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": custom_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"{selected_classifier} doesn't provide feature importance scores.")

    X_custom = filtered_df[feature_cols]
    y_pred_custom = model.predict(X_custom)
    filtered_df["Predicted_Phase"] = [label_map[i] for i in y_pred_custom]

    st.markdown("### Download Custom ENSO Prediction Results")
    st.download_button("📥 Download CSV", filtered_df.to_csv(index=False), "custom_enso_predictions.csv", mime="text/csv")

elif page == "🔮 Advanced Predictions":
    st.header("🔮 Advanced ENSO Predictions")

    # Introduction section
    st.markdown("""
    ### What is ENSO?
    The **El Niño-Southern Oscillation (ENSO)** is a climate pattern that affects weather worldwide. It has three phases:
    - 🔵 **La Niña**: Cooler ocean temperatures, often bringing more hurricanes and drought
    - ⚪ **Neutral**: Normal ocean temperatures and typical weather patterns  
    - 🔴 **El Niño**: Warmer ocean temperatures, often causing flooding and unusual weather
    
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

elif page == "💡 Model Insights":
    pass  # Remove this entire section
