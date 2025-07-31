# 🌊 ENSOcast — Decoding El Niño–Southern Oscillation

ENSOcast is an interactive climate prediction and visualization app that helps users understand, explore, and forecast ENSO (El Niño–Southern Oscillation) phases using real oceanic and atmospheric data. Built with [Streamlit](https://streamlit.io/), it offers a narrative-driven experience ideal for climate enthusiasts, educators, and researchers.

![ENSOcast Screenshot](screenshot.png) <!-- Replace with your actual screenshot -->

---

## 🔍 Features

- **🌍 Understand ENSO**  
  Learn about El Niño, La Niña, and Neutral phases and their global impacts.

- **🌡️ Ocean Temperatures**  
  Explore interactive sea surface temperature maps in the Niño 3.4 region using satellite datasets.

- **📊 Historical Pattern Explorer**  
  Visualize SST anomalies, ONI trends, SOI pressure changes, and seasonal ENSO behaviors from 1982–2024.

- **🔬 Model Performance**  
  Examine how a baseline Random Forest classifier performs using engineered climate features.

- **🛠️ Train Your Own Model**  
  Select ENSO phases and time periods, then train and evaluate your custom climate model.

---

## 🧠 How It Works

ENSOcast uses a Random Forest classifier trained on features like:

- SST Anomaly (Niño 3.4)
- Southern Oscillation Index (SOI)
- Lagged SST/SOI variables
- Seasonal signals encoded as sine/cosine

The model achieves ~82% accuracy in predicting monthly ENSO phases.

---

## 📦 Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/ensocast.git
cd ensocast
