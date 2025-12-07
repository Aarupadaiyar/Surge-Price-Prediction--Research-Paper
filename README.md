# Surge-Price-Prediction--Research-Paper
Title:

Machine Learning–Based Surge Price Prediction for Ride-Sharing Platforms Using Weather, Traffic, and Demand Data

Author:

Aarupadaiyaar KJ
School of Computer Science, Lovely Professional University

🚀 Project Overview

This research project investigates how surge pricing in ride-hailing platforms (Uber, Ola, Rapido, etc.) can be predicted using machine learning models. Surge pricing is influenced by real-time weather, traffic congestion, and demand levels.

The goal of this project is to:

Predict whether surge pricing will occur (classification)

Predict how much surge multiplier will apply (regression)

The dataset was created using real-world APIs, publicly available traffic data, and synthetic modeling, making it highly realistic and suitable for academic research.

📊 Dataset Description

The dataset combines:

1. Weather Data (API — Open-Meteo)

Fetched hourly for 7 major Indian cities:

Temperature

Relative humidity

Rainfall

2. Traffic Data (TomTom Congestion Index)

For each city:

Average congestion level

Peak congestion indicators

3. Fare Structure Data (Based on Ola/Uber pricing)

For each vehicle type:

Base fare per km

Minimum fare

Booking fee

Night multiplier

Vehicle types included:

Bike, Auto, Mini, Sedan, SUV

4. Synthetic Features

To realistically simulate ride-hailing patterns:

Distance distribution (70% short, 20% medium, 10% long)

Demand level based on time-of-day + weather

Surge multiplier modeled using rule-based and contextual factors

Surge flag (1 = surge, 0 = no surge)

🏗️ Feature Columns
Category	Variables
Weather	temperature_2m, relativehumidity_2m, rain
Time	hour, month, season
Traffic	congestion_level
Vehicle	vehicle_type
Fare	base_fare_per_km, minimum_fare, booking_fee, night_multiplier
Demand	demand_level
Target (Classification)	surge_flag
Target (Regression)	surge_multiplier
Final Output	total_fare

Dataset size: ~100,000 rows
Merged from weather + traffic + fare + synthetic ride-generation pipeline.

🧠 Machine Learning Models Used
Classification (Surge or No Surge)

LightGBM Classifier

XGBoost Classifier

Random Forest Classifier

Regression (Predict Surge Multiplier)

Extra Trees Regressor

XGBoost Regressor

Random Forest Regressor

These models were selected based on literature analysis and their performance on structured/tabular data.

🧪 Results Summary
Classification Models

Moderate accuracy (reflects real-world unpredictability of surge behavior)

Influenced most by demand, rainfall, and congestion

Regression Models

Achieved R² ≈ 1.0 due to deterministic fare logic

Extra Trees performed exceptionally well

Very low MAE and RMSE values

📁 Repository Structure (Recommended)
📦 Surge-Price-Prediction
 ┣ 📂 data/
 ┃ ┣ weather_data.csv
 ┃ ┣ traffic_data.csv
 ┃ ┣ fare_structure.csv
 ┃ ┗ final_dataset.csv
 ┣ 📂 notebooks/
 ┃ ┗ surge_prediction.ipynb
 ┣ 📂 models/
 ┃ ┣ classification_models.pkl
 ┃ ┗ regression_models.pkl
 ┣ 📄 README.md
 ┣ 📄 research_paper.pdf
 ┣ 📄 research_paper.docx
 ┗ 📄 requirements.txt

🔍 Research Paper Components

The IEEE-format research paper includes:

Abstract

Introduction

Literature Review

Data Collection Strategy

Methodology

ML Model Architecture

Results & Discussion

System Architecture Diagram

Error Analysis

Limitations

Future Work

References

⚙️ How to Run the Project
1. Install dependencies
pip install numpy pandas scikit-learn lightgbm xgboost requests

2. Run the main script
python surge_prediction.py

3. View results

Classification accuracy

Regression metrics

Feature importance chart

📌 Key Highlights of This Project

✔ Uses real APIs for weather
✔ Includes real traffic metrics from TomTom
✔ Fare structure based on public Ola/Uber data
✔ Realistic synthetic ride generation
✔ Full ML pipeline implemented
✔ IEEE-format research paper ready for submission
✔ Results are interpretable and academically strong

📌 Future Work

Future improvements may include:

Integrating real mobility datasets (Uber Movement, Google Traffic)

Building a deep-learning model (LSTM for time-series surge prediction)

Adding real supply-side data (number of active drivers)

Deploying a real-time surge prediction dashboard

📝 License

This project is for academic research and educational use.

🙌 Acknowledgements

Open-Meteo API

TomTom Traffic Index

Ola/Uber fare documentation

Lovely Professional University (LPU)
