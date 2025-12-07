# 🚖 Machine Learning–Based Surge Price Prediction  
### Using Weather, Traffic, and Demand Data

## **Author**
**Aarupadaiyaar KJ**  
School of Computer Science, Lovely Professional University  

---

# 📌 **Overview**

This project predicts **surge pricing** in ride-sharing platforms (Uber, Ola, Rapido) using **Machine Learning**.  
Two types of predictions are made:

1. **Classification** → Will surge happen? (0 or 1)  
2. **Regression** → How much surge multiplier will occur? (1.0 – 2.0)  

The dataset is built using **real APIs**, **traffic data**, and **synthetic modeling** to mimic real ride-hailing behavior in India.

---

# 📊 **Dataset Summary**

This project integrates data from:

### ✅ **1. Weather API — Open-Meteo**
- temperature_2m  
- relativehumidity_2m  
- rain  

### ✅ **2. Traffic Data — TomTom Index**
- congestion_level for 7 Indian cities

### ✅ **3. Fare Structure (Ola/Uber-like)**
Includes values for:
- base_fare_per_km  
- minimum_fare  
- booking_fee  
- night_multiplier  

Vehicle types:
```
Bike, Auto, Mini, Sedan, SUV
```

### ✅ **4. Synthetic Data Generation**
- distance_km (using realistic distribution)  
- demand_level (based on hour + rain)  
- surge_multiplier (behavior-based rules)  
- surge_flag (0 or 1)  

---

# 🧠 **Features Included**

| Category | Features |
|---------|----------|
| Time | hour, month, season |
| Weather | temperature_2m, humidity, rain |
| Traffic | congestion_level |
| Fare | base_fare_per_km, minimum_fare, booking_fee, night_multiplier |
| Vehicle | vehicle_type |
| Demand | demand_level |
| Target (Classification) | surge_flag |
| Target (Regression) | surge_multiplier |
| Output | total_fare |

Dataset size: **~100,000 rows**  

---

# 🏗️ **Project Structure**

```
📦 Surge-Price-Prediction
 ┣ 📂 data/
 ┃ ┣ weather_data.csv
 ┃ ┣ traffic_data.csv
 ┃ ┣ fare_data.csv
 ┃ ┗ final_dataset.csv
 ┣ 📂 models/
 ┃ ┣ classification/
 ┃ ┃ ┣ lightgbm_classifier.pkl
 ┃ ┃ ┣ xgboost_classifier.pkl
 ┃ ┃ ┗ random_forest_classifier.pkl
 ┃ ┗ regression/
 ┃   ┣ extratrees_regressor.pkl
 ┃   ┣ xgboost_regressor.pkl
 ┃   ┗ random_forest_regressor.pkl
 ┣ 📂 notebooks/
 ┃ ┗ surge_prediction.ipynb
 ┣ 📂 src/
 ┃ ┣ data_collection.py
 ┃ ┣ data_preprocessing.py
 ┃ ┣ feature_engineering.py
 ┃ ┣ model_training.py
 ┃ ┗ model_evaluation.py
 ┣ 📄 requirements.txt
 ┣ 📄 README.md
 ┣ 📄 research_paper.docx
 ┗ 📄 research_paper.pdf
```

---

# ⚙️ **Installation**

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ **Usage**

### **1. Run full pipeline**
```bash
python src/model_training.py
```

### **2. Train only classification models**
```bash
python src/model_training.py --task classification
```

### **3. Train only regression models**
```bash
python src/model_training.py --task regression
```

### **4. Open the notebook**
```
notebooks/surge_prediction.ipynb
```

---

# 🤖 **Machine Learning Models Used**

### **Classification Models**
- LightGBM Classifier  
- XGBoost Classifier  
- Random Forest Classifier  

### **Regression Models**
- Extra Trees Regressor  
- XGBoost Regressor  
- Random Forest Regressor  

---

# 🧪 **Results Summary**

### **Classification**
- Moderate performance  
- Surge prediction accuracy depends on demand, rainfall & traffic  

### **Regression**
- **R² Score ≈ 1.0**  
- Surge multiplier prediction extremely accurate  
- Extra Trees & Random Forest perform best  

---

# 📈 **Feature Importance**

Top predictors include:
- demand_level  
- congestion_level  
- rain  
- distance_km  
- vehicle_type  
- night_multiplier  
- base_fare_per_km  

---

# 📝 **Research Paper**
Full IEEE-format research paper included:  
```
research_paper.docx  
research_paper.pdf
```

---

# 🔮 **Future Work**
- Use LSTM or Temporal ML for time-series surge prediction  
- Integrate real Uber Movement or Google mobility data  
- Build a live dashboard for real-time surge prediction  

---

# 📜 **License**
This project is for research and educational purposes only.

---

# 🙏 **Acknowledgements**
- Open-Meteo API  
- TomTom Traffic Index  
- Ola/Uber fare charts  
- Lovely Professional University  

---

