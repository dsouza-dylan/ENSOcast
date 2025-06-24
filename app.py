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
    ["🌡 Global SST Snapshot", "📈 Historical Trends", "💡 Model Insights", "🛠 Interactive Prediction Tool", "🔮 Advanced Predictions"],
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

elif page == "🛠 Interactive Prediction Tool":
    st.header("🛠 Interactive Prediction Tool")
    st.markdown("### Try the Model Yourself")

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
    st.header("🔮 Advanced Predictions & Analysis")

    # Section 1: Specific Date Prediction with Probabilities
    st.markdown("### 🎯 Predict ENSO Phase for Specific Date")

    col1, col2 = st.columns(2)
    with col1:
        target_year = st.number_input("Year", min_value=1982, max_value=2030, value=2024)
    with col2:
        target_month = st.selectbox("Month", [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ], index=0)

    month_num = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }[target_month]

    target_date = datetime.date(target_year, month_num, 1)

    if st.button("🔮 Make Prediction"):
        try:
            # Create features for the target date
            X_target = create_feature_for_date(target_date, df, feature_cols)

            # Get prediction and probabilities
            prediction = model.predict(X_target)[0]
            probabilities = model.predict_proba(X_target)[0]

            predicted_phase = label_map[prediction]

            # Display results
            st.success(f"**Predicted ENSO Phase for {target_month} {target_year}: {predicted_phase}**")

            # Show probabilities
            st.markdown("### 📊 Prediction Probabilities")
            prob_df = pd.DataFrame({
                "ENSO Phase": ["La Niña", "Neutral", "El Niño"],
                "Probability": probabilities,
                "Percentage": [f"{p*100:.1f}%" for p in probabilities]
            })

            # Create probability bar chart
            fig_prob = px.bar(prob_df, x="ENSO Phase", y="Probability",
                            text="Percentage", color="ENSO Phase",
                            color_discrete_map={
                                "La Niña": "blue",
                                "Neutral": "gray",
                                "El Niño": "red"
                            })
            fig_prob.update_traces(textposition='outside')
            fig_prob.update_layout(showlegend=False, yaxis_title="Probability")
            st.plotly_chart(fig_prob, use_container_width=True)

            # Confidence assessment
            max_prob = max(probabilities)
            if max_prob > 0.7:
                confidence = "High"
                confidence_color = "green"
            elif max_prob > 0.5:
                confidence = "Medium"
                confidence_color = "orange"
            else:
                confidence = "Low"
                confidence_color = "red"

            st.markdown(f"**Model Confidence**: :{confidence_color}[{confidence}] ({max_prob*100:.1f}%)")

        except Exception as e:
            st.error(f"Error making prediction: {e}")

    st.markdown("---")

    # Section 2: Predicted vs Actual Timeline
    st.markdown("### 📈 Predicted vs Actual ENSO Phases Timeline")

    # Filter controls
    timeline_years = st.slider("Timeline Year Range", 1982, 2025, (2010, 2020), key="timeline")

    df_timeline = df[
        (df["Date"].dt.year >= timeline_years[0]) &
        (df["Date"].dt.year <= timeline_years[1])
    ].copy()

    # Create numerical mapping for plotting
    phase_to_num = {"La Niña": -1, "Neutral": 0, "El Niño": 1}
    df_timeline["True_Phase_Num"] = df_timeline["True_Phase"].map(phase_to_num)
    df_timeline["Predicted_Phase_Num"] = df_timeline["Predicted_Phase"].map(phase_to_num)

    # Create timeline plot
    fig_timeline = go.Figure()

    # Add actual phases
    fig_timeline.add_trace(go.Scatter(
        x=df_timeline["Date"],
        y=df_timeline["True_Phase_Num"],
        mode='lines+markers',
        name='Actual Phase',
        line=dict(color='blue', width=3),
        marker=dict(size=6)
    ))

    # Add predicted phases
    fig_timeline.add_trace(go.Scatter(
        x=df_timeline["Date"],
        y=df_timeline["Predicted_Phase_Num"],
        mode='lines+markers',
        name='Predicted Phase',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=4)
    ))

    fig_timeline.update_layout(
        title="ENSO Phase Predictions vs Reality",
        xaxis_title="Date",
        yaxis_title="ENSO Phase",
        yaxis=dict(
            tickmode='array',
            tickvals=[-1, 0, 1],
            ticktext=['La Niña', 'Neutral', 'El Niño']
        ),
        template="plotly_dark",
        hovermode='x unified'
    )

    # Add reference lines
    fig_timeline.add_hline(y=0.5, line_dash="dash", line_color="orange", opacity=0.3)
    fig_timeline.add_hline(y=-0.5, line_dash="dash", line_color="orange", opacity=0.3)

    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("---")

    # Section 3: SHAP Analysis (Collapsible)
    if SHAP_AVAILABLE:
        with st.expander("🧠 SHAP Feature Explanation Analysis", expanded=False):
            st.markdown("### Understanding Model Decisions with SHAP")
            st.info("SHAP (SHapley Additive exPlanations) shows how each feature contributes to individual predictions.")

            # Sample selection for SHAP
            sample_size = min(100, len(df))
            sample_indices = np.random.choice(len(df), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]

            with st.spinner("Calculating SHAP values... This may take a moment."):
                try:
                    explainer, shap_values = prepare_shap_values(model, X_sample)

                    if shap_values is not None:
                        # SHAP Summary Plot
                        st.markdown("#### SHAP Summary Plot")
                        fig_shap, ax = plt.subplots(figsize=(10, 6))

                        # For multiclass, we'll show the summary for the El Niño class (index 2)
                        shap.summary_plot(shap_values[2], X_sample, feature_names=feature_cols,
                                        show=False, plot_type="bar")
                        st.pyplot(fig_shap)

                        st.markdown("#### SHAP Feature Impact")
                        fig_shap2, ax2 = plt.subplots(figsize=(10, 8))
                        shap.summary_plot(shap_values[2], X_sample, feature_names=feature_cols, show=False)
                        st.pyplot(fig_shap2)

                        st.success("✅ SHAP analysis complete! This shows which features most influence El Niño predictions.")

                except Exception as e:
                    st.error(f"Error calculating SHAP values: {e}")
                    st.info("SHAP analysis requires compatible model types and sufficient memory.")
    else:
        st.info("💡 Install SHAP (`pip install shap`) to unlock advanced feature explanation capabilities!")

    st.markdown("---")

    # Section 4: Advanced Model Comparison
    st.markdown("### 🏆 Advanced Model Performance Comparison")

    comparison_years = st.slider("Comparison Year Range", 1982, 2025, (2000, 2020), key="comparison")

    df_comparison = df[
        (df["Date"].dt.year >= comparison_years[0]) &
        (df["Date"].dt.year <= comparison_years[1])
    ]

    X_comp = df_comparison[feature_cols]
    y_comp = df_comparison["True_Phase"]

    if st.button("🚀 Run Model Comparison"):
        models_to_compare = {
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "SVM": SVC(random_state=42, probability=True),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
        }

        comparison_results = []

        progress_bar = st.progress(0)
        for i, (name, model_comp) in enumerate(models_to_compare.items()):
            with st.spinner(f"Training {name}..."):
                # Cross-validation
                cv_scores = cross_val_score(model_comp, X_comp, y_comp, cv=5, scoring='accuracy')

                comparison_results.append({
                    "Model": name,
                    "CV Mean Accuracy": cv_scores.mean(),
                    "CV Std Dev": cv_scores.std(),
                    "CV Score Range": f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
                })

                progress_bar.progress((i + 1) / len(models_to_compare))

        # Display results
        results_df = pd.DataFrame(comparison_results)
        st.dataframe(results_df.round(4))

        # Visualization
        fig_comp = px.bar(results_df, x="Model", y="CV Mean Accuracy",
                         error_y="CV Std Dev",
                         title="Model Performance Comparison")
        fig_comp.update_layout(yaxis_title="Cross-Validation Accuracy")
        st.plotly_chart(fig_comp, use_container_width=True)

        # Best model recommendation
        best_model = results_df.loc[results_df["CV Mean Accuracy"].idxmax(), "Model"]
        best_accuracy = results_df.loc[results_df["CV Mean Accuracy"].idxmax(), "CV Mean Accuracy"]
        st.success(f"🏆 Best performing model: **{best_model}** with {best_accuracy:.1%} accuracy")
