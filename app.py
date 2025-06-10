import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import classification_report, confusion_matrix

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
    ["🌡 Global SST Snapshot", "📈 Historical Trends", "🛠 Interactive Prediction Tool"],
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

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

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

elif page == "🛠 Interactive Prediction Tool":
    st.header("🛠 Interactive Prediction Tool")
    from sklearn.metrics import accuracy_score
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

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.3, shuffle=False)

    from sklearn.ensemble import RandomForestClassifier
    custom_model = RandomForestClassifier(random_state=42)
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

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": custom_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    X_custom = filtered_df[feature_cols]
    y_pred_custom = model.predict(X_custom)
    filtered_df["Predicted_Phase"] = [label_map[i] for i in y_pred_custom]


    st.markdown("### Download Custom ENSO Prediction Results")
    st.download_button("📥 Download CSV", filtered_df.to_csv(index=False), "custom_enso_predictions.csv", mime="text/csv")
