# 🏭 Stenter (Ramöz) Machine AI Decision Support & Optimization System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-MLOps-red?style=for-the-badge&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data_Engineering-green?style=for-the-badge&logo=pandas)

An end-to-end **Industry 4.0** Machine Learning (MLOps) and Prescriptive Analytics project developed for Stenter (Ramöz) machines, one of the most energy-intensive equipment in the textile industry. By processing data from over 120 IoT SCADA sensors, a "Digital Twin" of the machine was created, and an algorithmic Decision Support System was deployed, achieving a **net 4.96% savings in natural gas consumption** without any physical hardware modifications.

---

## 🌟 Key Features

*   📊 **Digital Twin:** Modeled the specific natural gas consumption of the machine with an **84.18% R²** accuracy rate using a Random Forest Regressor.
*   🧠 **Prescriptive Analytics:** Dynamically calculates the "Optimal Machine Recipe" using SciPy (Powell Optimization) to minimize energy consumption without reducing production speed.
*   🛠️ **Predictive Maintenance (PdM):** Detects hardware insulation leaks and thermal breakdown risks across 10 different heating cabins via sensor data, utilizing the Steady-State Error principle.
*   📈 **Explainable AI (XAI):** Integrated SHAP to demystify the "Black Box" nature of the model, mathematically proving the directional impact of machine settings (fan speeds, temperatures) on energy bills.
*   💻 **MLOps Dashboard:** Developed an interactive web interface with Streamlit and Session State architecture, enabling factory operators to run real-time cost/benefit simulations and receive quality control alerts.

---

## 🏗️ Project Architecture & Data Pipeline

The project is designed in 5 main phases conforming to software engineering standards:
1. **Data Engineering:** Cleaned raw `.rpt` logs using Regex, removed BOM characters, and converted them into high-performance `.parquet` format for memory optimization.
2. **Signal Processing:** Applied 10-minute Time Series Resampling via Pandas to dampen sensor latency.
3. **Outlier Filtering:** Filtered out noisy production regimes using the statistical IQR (Interquartile Range) method.
4. **Model Training:** Trained tree-based ensemble models with Scikit-learn and performed hyperparameter optimizations.
5. **Deployment:** Delivered the model to end-users utilizing Streamlit Caching and Two-Way Binding architectures.

---

## 📂 Folder Structure

```text
Ramoz-AI-Optimizer/
├── app/
│   └── app.py                         # Streamlit MLOps Decision Support Dashboard
├── assets/                            # Charts, SHAP plots, and UI screenshots
├── data/                              # Cleaned datasets in .parquet format
├── src/                               # Machine Learning Pipeline Scripts
│   ├── 01_data_preprocessing.py       # Data cleaning and Parquet conversion
│   ├── 02_feature_engineering.py      # Resampling, IQR filtering, and Random Forest model training
│   ├── 03_correlation_analysis.py     # Pearson correlation and regression analysis
│   ├── 04_shap_analysis.py            # Explainable AI (XAI) analysis
│   ├── 05_predictive_maintenance.py   # Thermal stress and fault detection algorithm
│   └── 06_optimization_engine.py      # SciPy Powell optimization engine
├── requirements.txt                   # Project dependencies and library versions
└── README.md
