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
# tab1, tab2, tab3, tab4 = st.tabs(["🌡 SST Snapshot", "📈 Trends", "🔎 Model Insights", "📤 Download"])
st.sidebar.title("📂 ENSOcast Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🌡 SST Snapshot", "📈 Trends", "🔎 Model Insights", "📤 Download"],
    index=0
)
# --- Tab 1: SST Snapshot ---
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
        fig, ax = plt.subplots(figsize=(12, 4))
        sst_slice.plot(ax=ax, cmap='coolwarm', cbar_kwargs={"label": "°C"})
        ax.add_patch(patches.Rectangle((190, -5), 50, 10, edgecolor='black', facecolor='none', linewidth=2))
        ax.text(192, 6, 'Niño 3.4 Region', color='black')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to fetch SST data for {selected_month} {selected_year}. Error: {e}")

elif page == "📈 Trends":
    st.header("📈 Historical Trends")

    # --- User Filters ---
    years = st.slider("Select Year Range", 1982, 2025, (2000, 2020))
    selected_phases = st.multiselect(
        "Select ENSO Phases", ["El Niño", "La Niña", "Neutral"],
        default=["El Niño", "La Niña", "Neutral"]
    )

    # --- Apply Filters ---
    df_filtered = df[
        (df["Date"].dt.year >= years[0]) &
        (df["Date"].dt.year <= years[1]) &
        (df["ENSO_Phase"].isin(selected_phases))
    ]

    # --- SST Anomaly vs Absolute SST ---
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    st.markdown("### SST Anomaly vs Absolute SST (Dual Axis)")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"], y=df_filtered["SST_Climatology"],
            name="SST Climatology (°C)",
            line=dict(color='deepskyblue', dash='dot')
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"], y=df_filtered["SST"],
            name="SST (°C)",
            line=dict(color='orange')
        ),
        secondary_y=False,
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="SST (°C)",
        legend=dict(x=0.01, y=1.1),
        template="plotly_dark",
        margin=dict(t=30, b=30)
    )

    fig.update_yaxes(title_text="SST (°C)", range=[24, 30], secondary_y=False)
    fig.update_yaxes(title_text="Climatology (°C)", range=[24, 30], secondary_y=True, showticklabels=False)

    climatology_min = df_filtered["SST_Climatology"].min()
    climatology_max = df_filtered["SST_Climatology"].max()

    fig.add_hline(y=climatology_min, line_dash="dot", line_color="gray",
                  annotation_text="Climatology Min", annotation_position="bottom left")

    fig.add_hline(y=climatology_max, line_dash="dot", line_color="gray",
                  annotation_text="Climatology Max", annotation_position="top left")

    st.plotly_chart(fig, use_container_width=True)

    # --- ONI Timeline ---
    st.markdown("### ONI Timeline")
    fig_oni = px.line(df_filtered, x="Date", y="ONI", title="ONI (Oceanic Niño Index) Over Time")
    fig_oni.add_hline(y=0.5, line_dash="dot", line_color="red",
                      annotation_text="El Niño Threshold", annotation_position="bottom right")
    fig_oni.add_hline(y=-0.5, line_dash="dot", line_color="blue",
                      annotation_text="La Niña Threshold", annotation_position="top right")

    st.plotly_chart(fig_oni, use_container_width=True)


# # --- Tab 2: Trends ---
# elif page == "📈 Trends":
#     st.header("📈 Historical Trends")
#     # st.markdown("### SST Anomaly Timeline")
#     # fig = px.line(df, x="Date", y="SST_Anomaly", labels={"SST_Anomaly": "SST Anomaly (°C)"})
#     # st.plotly_chart(fig, use_container_width=True)
#     #
#     # st.markdown("### SST Anomaly Timeline")
#     # fig_abs = px.line(df, x="Date", y="SST", labels={"SST": "SST (°C)"})
#     # st.plotly_chart(fig_abs, use_container_width=True)
#
#     from plotly.subplots import make_subplots
#     import plotly.graph_objects as go
#
#     st.markdown("### SST Anomaly vs Absolute SST (Dual Axis)")
#
#     fig = make_subplots(specs=[[{"secondary_y": True}]])
#
#     # Add SST Anomaly
#     fig.add_trace(
#         go.Scatter(x=df["Date"], y=df["SST_Climatology"], name="SST Climatology (°C)", line=dict(color='deepskyblue', dash = 'dot')),
#         secondary_y=True,
#     )
#
#     # Add Absolute SST
#     fig.add_trace(
#         go.Scatter(x=df["Date"], y=df["SST"], name="SST (°C)", line=dict(color='orange')),
#         secondary_y=False,
#     )
#
#     fig.update_layout(
#         xaxis_title="Date",
#         yaxis_title="SST (°C)",
#         legend=dict(x=0.01, y=1.1),
#         template="plotly_dark",
#         margin=dict(t=30, b=30)
#     )
#
#     fig.update_yaxes(title_text="SST (°C)", range=[24, 30], secondary_y=False)
#     fig.update_yaxes(range=[24, 30], secondary_y=True, showticklabels = False)
#
#     climatology_min = df["SST_Climatology"].min()
#     climatology_max = df["SST_Climatology"].max()
#
#     fig.add_hline(y=climatology_min, line_dash="dot", line_color="gray",
#                   annotation_text="Climatology Min", annotation_position="bottom left")
#
#     fig.add_hline(y=climatology_max, line_dash="dot", line_color="gray",
#                   annotation_text="Climatology Max", annotation_position="top left")
#     st.plotly_chart(fig, use_container_width=True)
#
#
#
#     st.markdown("### ONI Timeline")
#     fig_oni = px.line(df, x="Date", y="ONI", title="ONI (Oceanic Niño Index) Over Time", labels={"oni": "ONI Value"})
#     fig_oni.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="El Niño Threshold", annotation_position="bottom right")
#     fig_oni.add_hline(y=-0.5, line_dash="dot", line_color="blue", annotation_text="La Niña Threshold", annotation_position="top right")
#     st.plotly_chart(fig_oni, use_container_width=True)

    # st.markdown("### Predicted ENSO Phase")
    # fig2 = px.line(df, x="Date", y="Predicted_Phase", color_discrete_sequence=["#e74c3c", "#3498db", "#95a5a6"])
    # st.plotly_chart(fig2, use_container_width=True)

    # Map ENSO phases to values
    # enso_label_map = {"El Niño": 1, "La Niña": -1, "Neutral": 0}
    # df["ENSO_Num"] = df["ENSO_Label"].map(enso_label_map)
    # import plotly.graph_objects as go
    #
    # st.markdown("### ENSO Phase Timeline")
    #
    # figx = go.Figure()
    #
    # figx.add_trace(go.Scatter(
    #     x=df["Date"],
    #     y=df["ENSO_Num"],
    #     mode="lines",
    #     line=dict(shape="hv", width=0),  # Step line with filled areas only
    #     fill="tozeroy",
    #     fillcolor="rgba(231, 76, 60, 0.6)",  # Red for El Niño
    #     name="El Niño",
    #     hoverinfo="skip",
    #     showlegend=False
    # ))
    #
    # figx.add_trace(go.Scatter(
    #     x=df["Date"],
    #     y=[-1 if p == "La Niña" else None for p in df["ENSO_Label"]],
    #     mode="markers",
    #     marker=dict(color="#3498db", size=4),
    #     name="La Niña"
    # ))
    #
    # figx.add_trace(go.Scatter(
    #     x=df["Date"],
    #     y=[0 if p == "Neutral" else None for p in df["ENSO_Label"]],
    #     mode="markers",
    #     marker=dict(color="#95a5a6", size=4),
    #     name="Neutral"
    # ))
    #
    # figx.update_layout(
    #     yaxis=dict(
    #         tickvals=[-1, 0, 1],
    #         ticktext=["La Niña", "Neutral", "El Niño"],
    #         title="ENSO Phase"
    #     ),
    #     xaxis_title="Date",
    #     template="plotly_dark",
    #     showlegend=True
    # )
    #
    # st.plotly_chart(figx, use_container_width=True)


# --- Tab 3: Model Insights ---
elif page == "🔎 Model Insights":
    st.header("🔎 Model Insights")
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(df["True_Phase"], df["Predicted_Phase"])
    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

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

    st.markdown("### Download Predictions CSV")
    st.download_button("📥 Download ENSO Predictions", data=df.to_csv(index=False), file_name="enso_predictions.csv", mime="text/csv")

# --- Tab 4: Download ---
elif page == "📤 Download":
    from sklearn.metrics import accuracy_score
    st.header("📤 Download Center")
    st.markdown("### Custom Model Evaluation")

    # --- User Filters ---
    years = st.slider("Select Year Range", 1982, 2025, (2000, 2020))
    selected_phases = st.multiselect("Select ENSO Phases", ["El Niño", "La Niña", "Neutral"], default=["El Niño", "La Niña", "Neutral"])

    filtered_df = df[
        (df["Date"].dt.year >= years[0]) &
        (df["Date"].dt.year <= years[1]) &
        (df["True_Phase"].isin(selected_phases))
    ]

    # --- Prepare Features & Labels ---
    X_custom = filtered_df[feature_cols]
    y_custom = filtered_df["True_Phase"]

    # --- Train/Test Split ---
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.3, shuffle=False)

    # --- Train Custom Model ---
    from sklearn.ensemble import RandomForestClassifier
    custom_model = RandomForestClassifier(random_state=42)
    custom_model.fit(X_train, y_train)
    y_pred_custom = custom_model.predict(X_test)

    # --- Accuracy ---
    custom_accuracy = accuracy_score(y_test, y_pred_custom)
    st.metric("Custom Model Accuracy", f"{custom_accuracy * 100:.2f}%")

    st.markdown("### Classification Report")
    report = classification_report(y_test, y_pred_custom, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_custom, labels=["El Niño", "La Niña", "Neutral"])
    st.dataframe(pd.DataFrame(cm, index=["True El Niño", "True La Niña", "True Neutral"],
                              columns=["Pred El Niño", "Pred La Niña", "Pred Neutral"]))

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": custom_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    X_custom = filtered_df[feature_cols]
    y_pred_custom = model.predict(X_custom)
    filtered_df["Predicted_Phase"] = [label_map[i] for i in y_pred_custom]


    st.markdown("### Download Custom Predictions CSV")
    st.download_button("📥 Download Custom Predictions", filtered_df.to_csv(index=False), "custom_enso_predictions.csv", mime="text/csv")
