# 🌊 ENSOcast

**Decoding El Niño–Southern Oscillation with Machine Learning**

A comprehensive Streamlit web application that leverages machine learning to analyze, visualize, and predict ENSO (El Niño-Southern Oscillation) patterns using historical climate data.

![ENSOcast Demo](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌍 What is ENSO?

The El Niño-Southern Oscillation (ENSO) is one of the most influential climate patterns on Earth, affecting weather conditions globally. ENSOcast helps users understand this complex phenomenon through:

- **Interactive visualizations** of ocean temperature data
- **Machine learning predictions** of ENSO phases
- **Historical pattern analysis** spanning 40+ years of climate data
- **Educational content** explaining ENSO's global impacts

## ✨ Features

### 🎓 Educational Dashboard
- **Understanding ENSO**: Interactive introduction to El Niño, La Niña, and Neutral phases
- **Global Impact Visualization**: Explore how ENSO affects agriculture, weather, economy, and marine life

### 🌡️ Real-Time Ocean Analysis
- **Live SST Data**: Connect to NOAA's satellite data for current sea surface temperatures
- **Niño 3.4 Region Focus**: Visualize the critical Pacific region that drives ENSO
- **Interactive Date Selection**: Explore historical ocean conditions from 1982-2024

### 📊 Comprehensive Data Exploration
- **Multi-variable Analysis**: SST anomalies, Southern Oscillation Index (SOI), and Oceanic Niño Index (ONI)
- **Seasonal Pattern Recognition**: Discover when El Niño and La Niña events typically occur
- **Customizable Time Ranges**: Filter data by years and ENSO phases

### 🔬 Machine Learning Models
- **Baseline Random Forest Model**: Pre-trained classifier achieving 82% accuracy
- **Feature Engineering**: Lagged variables, seasonal encoding, and anomaly calculations
- **Performance Metrics**: Detailed confusion matrices and classification reports

### 🛠️ Custom Model Training
- **Interactive ML Pipeline**: Train your own ENSO prediction models
- **Hyperparameter Control**: Experiment with different time periods and phase combinations
- **Comparative Analysis**: Compare custom models against the baseline

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ensocast.git
cd ensocast
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare the data:**
```bash
# Ensure merged_enso.csv is in the project root
# Download from: [Your data source URL]
```

4. **Run the application:**
```bash
streamlit run ensocastapp.py
```

5. **Open your browser:**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
ensocast/
├── ensocastapp.py          # Main Streamlit application
├── merged_enso.csv         # Historical ENSO dataset
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── assets/                # Images and static files
    └── screenshots/       # App screenshots
```

## 📊 Data Sources

ENSOcast integrates multiple authoritative climate datasets:

- **Sea Surface Temperature**: NOAA OISST v2.1 High Resolution Dataset
- **Southern Oscillation Index**: Australian Bureau of Meteorology
- **Oceanic Niño Index**: NOAA Climate Prediction Center
- **ENSO Classifications**: Based on ONI thresholds (±0.5°C)

### Data Features

| Feature | Description | Source |
|---------|-------------|---------|
| `SST` | Sea Surface Temperature (°C) | NOAA OISST |
| `SST_Anomaly` | Temperature deviation from climatology | Derived |
| `SOI` | Southern Oscillation Index | Australian BOM |
| `ONI` | Oceanic Niño Index | NOAA CPC |
| `ENSO_Phase` | El Niño/Neutral/La Niña classification | ONI-based |

## 🤖 Machine Learning Pipeline

### Feature Engineering

ENSOcast applies sophisticated preprocessing to capture ENSO dynamics:

```python
# Temporal lag features (1-3 months)
SST_Anomaly_lag_1, SST_Anomaly_lag_2, SST_Anomaly_lag_3
SOI_lag_1, SOI_lag_2, SOI_lag_3

# Seasonal encoding
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Features**: 10 engineered climate indicators  
- **Target**: 3-class ENSO phase classification
- **Validation**: Time-series split (80% train, 20% test)
- **Performance**: 82% accuracy on held-out data

### Model Interpretability

- **Feature Importance Analysis**: Identify key climate drivers
- **Confusion Matrix**: Understand phase-specific performance
- **Classification Reports**: Precision, recall, and F1-scores per phase

## 🎯 Use Cases

### 🎓 Education
- **Climate Science Students**: Interactive learning tool for ENSO concepts
- **Researchers**: Rapid prototyping of ENSO prediction models
- **Educators**: Visual demonstrations of climate oscillations

### 📈 Research & Analysis
- **Pattern Discovery**: Identify long-term ENSO trends and cycles
- **Model Comparison**: Benchmark different ML approaches
- **Feature Analysis**: Understand climate variable relationships

### 🌐 Practical Applications
- **Agricultural Planning**: Seasonal forecasting for crop management
- **Risk Assessment**: Extreme weather preparation
- **Climate Monitoring**: Real-time ENSO phase tracking

## 🛠️ Technical Stack

### Core Technologies
- **Frontend**: Streamlit (Interactive web framework)
- **Backend**: Python 3.8+
- **Machine Learning**: scikit-learn, Random Forest
- **Data Processing**: pandas, numpy, xarray
- **Visualization**: plotly, matplotlib

### Key Dependencies

```python
streamlit>=1.28.0        # Web application framework
pandas>=1.5.0           # Data manipulation
numpy>=1.24.0           # Numerical computing
scikit-learn>=1.3.0     # Machine learning
plotly>=5.15.0          # Interactive visualizations
xarray>=2023.1.0        # Multi-dimensional arrays
matplotlib>=3.7.0       # Statistical plotting
joblib>=1.3.0           # Model serialization
```

## 📸 Screenshots

### Dashboard Overview
![Dashboard](assets/screenshots/dashboard.png)

### Ocean Temperature Analysis
![Ocean Temps](assets/screenshots/ocean_temps.png)

### Model Performance
![Model Performance](assets/screenshots/model_performance.png)

## 🤝 Contributing

We welcome contributions to ENSOcast! Here's how you can help:

### Development Setup

1. **Fork the repository**
2. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes**
4. **Add tests** (if applicable)
5. **Submit a pull request**

### Contribution Guidelines

- **Code Style**: Follow PEP 8 conventions
- **Documentation**: Update README and docstrings
- **Testing**: Ensure existing functionality isn't broken
- **Performance**: Consider computational efficiency for large datasets

### Areas for Contribution

- 🌐 **Additional Data Sources**: Integrate more climate indices
- 🤖 **Advanced Models**: Implement deep learning approaches
- 📱 **Mobile Optimization**: Improve responsive design
- 🌍 **Internationalization**: Multi-language support
- 📊 **New Visualizations**: Creative data representations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NOAA** for providing comprehensive climate datasets
- **Australian Bureau of Meteorology** for SOI data
- **Streamlit Community** for the excellent web framework
- **Climate Research Community** for ENSO insights and methodologies

## 📞 Contact

**Dylan Dsouza** - *Creator & Maintainer*

- 📧 Email: [your.email@example.com]
- 🐙 GitHub: [@yourusername](https://github.com/yourusername)
- 💼 LinkedIn: [Your LinkedIn Profile]

## 🔗 Related Projects

- [Climate Data Analysis Tools](https://github.com/climate-tools)
- [ENSO Prediction Models](https://github.com/enso-ml)
- [Ocean Temperature Visualizations](https://github.com/ocean-viz)

## 📚 References

1. Trenberth, K. E. (1997). The definition of El Niño. *Bulletin of the American Meteorological Society*, 78(12), 2771-2777.
2. McPhaden, M. J., et al. (2006). ENSO as an integrating concept in Earth science. *Science*, 314(5806), 1740-1745.
3. Bjerknes, J. (1969). Atmospheric teleconnections from the equatorial Pacific. *Monthly Weather Review*, 97(3), 163-172.

---

<div align="center">

**🌊 ENSOcast - Decoding El Niño–Southern Oscillation**

*Making climate science accessible through interactive visualization and machine learning*

[⭐ Star this repository](https://github.com/yourusername/ensocast) | [🐛 Report Bug](https://github.com/yourusername/ensocast/issues) | [💡 Request Feature](https://github.com/yourusername/ensocast/issues)

</div>
