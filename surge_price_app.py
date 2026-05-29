
# ─────────────────────────────────────────────────────────────────────────────
#  SURGE PRICE PREDICTION — STREAMLIT APP
#  Built by Aarav | Run in Google Colab with:
#    !pip install streamlit pyngrok lightgbm xgboost -q
#    Then run the last cell to launch
# ─────────────────────────────────────────────────────────────────────────────

import requests, pandas as pd, numpy as np, streamlit as st, joblib, os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Surge Price Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --card: #1a1a26;
    --border: #2a2a3d;
    --accent: #f5a623;
    --accent2: #e84855;
    --green: #39d353;
    --text: #e8e8f0;
    --muted: #7070a0;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Header */
.header-block {
    background: linear-gradient(135deg, #1a1a26 0%, #0f0f1a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.header-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.header-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #f5a623, #e84855);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    margin: 4px 0;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* Result box */
.result-surge {
    background: linear-gradient(135deg, #2a1215, #1a0a0a);
    border: 2px solid var(--accent2);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
}
.result-no-surge {
    background: linear-gradient(135deg, #0f2a15, #0a1a0a);
    border: 2px solid var(--green);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
}
.result-icon { font-size: 3rem; margin-bottom: 8px; }
.result-title { font-size: 1.8rem; font-weight: 800; margin: 0; }
.result-detail {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 8px;
}

/* Fare box */
.fare-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 16px;
}
.fare-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
}
.fare-row:last-child { border-bottom: none; font-weight: 700; font-size: 1rem; }
.fare-label { color: var(--muted); }
.fare-value { color: var(--text); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 0 !important;
    width: 100% !important;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Section labels */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 20px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Progress / accuracy bars */
.acc-bar-bg {
    background: var(--border);
    border-radius: 4px;
    height: 8px;
    width: 100%;
    margin-top: 4px;
}
.acc-bar-fill {
    border-radius: 4px;
    height: 8px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS ────────────────────────────────────────────────────────────────
CITIES = {
    "Bangalore": (12.9716, 77.5946),
    "Chennai":   (13.0827, 80.2707),
    "Mumbai":    (19.0760, 72.8777),
    "Delhi":     (28.7041, 77.1025),
    "Hyderabad": (17.3850, 78.4867),
    "Pune":      (18.5204, 73.8567),
    "Kolkata":   (22.5726, 88.3639)
}

TRAFFIC = {
    "Bangalore": 51, "Delhi": 43, "Mumbai": 47,
    "Chennai": 32,   "Hyderabad": 38, "Pune": 33, "Kolkata": 28
}

FARE_DATA = {
    "Bangalore": {"Bike":7,"Auto":11,"Mini":15,"Sedan":18,"SUV":25},
    "Chennai":   {"Bike":6,"Auto":10,"Mini":14,"Sedan":17,"SUV":24},
    "Mumbai":    {"Bike":7,"Auto":12,"Mini":17,"Sedan":20,"SUV":28},
    "Delhi":     {"Bike":6,"Auto":10,"Mini":14,"Sedan":17,"SUV":23},
    "Hyderabad": {"Bike":6,"Auto":10,"Mini":14,"Sedan":17,"SUV":23},
    "Pune":      {"Bike":7,"Auto":11,"Mini":16,"Sedan":19,"SUV":26},
    "Kolkata":   {"Bike":6,"Auto":10,"Mini":14,"Sedan":17,"SUV":23}
}

MIN_FARE = {"Bike":25,"Auto":35,"Mini":50,"Sedan":65,"SUV":90}
BOOKING_FEE = {"Bike":10,"Auto":20,"Mini":30,"Sedan":35,"SUV":40}
NIGHT_MULT = {"Bangalore":1.5,"Chennai":1.5,"Mumbai":1.5,"Delhi":1.5,
              "Hyderabad":1.5,"Pune":1.5,"Kolkata":1.25}

MODEL_PATH_CLS = "/content/surge_classifier.pkl"
MODEL_PATH_REG = "/content/surge_regressor.pkl"
ENCODER_PATH   = "/content/surge_encoders.pkl"


# ─── DATA + TRAINING ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_weather():
    frames = []
    for city, (lat, lon) in CITIES.items():
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,relativehumidity_2m,rain",
                "timezone": "Asia/Kolkata"
            }
            data = requests.get(url, params=params, timeout=10).json()
            df = pd.DataFrame(data["hourly"])
            df["city"] = city
            frames.append(df)
        except:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def to_season(m):
    if m in [12,1,2]: return "Winter"
    if m in [3,4,5]:  return "Summer"
    if m in [6,7,8,9]: return "Monsoon"
    return "Post-Monsoon"


@st.cache_resource(show_spinner=False)
def build_and_train():
    """Build dataset, train models, cache everything."""

    with st.spinner("⚙️ Fetching live weather data..."):
        weather_df = fetch_weather()

    weather_df["time"]  = pd.to_datetime(weather_df["time"])
    weather_df["hour"]  = weather_df["time"].dt.hour
    weather_df["month"] = weather_df["time"].dt.month
    weather_df["season"] = weather_df["month"].apply(to_season)

    traffic_df = pd.DataFrame({
        "city": list(TRAFFIC.keys()),
        "congestion_level": list(TRAFFIC.values())
    })

    merged_df = weather_df.merge(traffic_df, on="city", how="left")

    fare_rows = []
    for city, vmap in FARE_DATA.items():
        for vtype, bpk in vmap.items():
            fare_rows.append({
                "city": city, "vehicle_type": vtype,
                "base_fare_per_km": bpk,
                "minimum_fare": MIN_FARE[vtype],
                "booking_fee": BOOKING_FEE[vtype],
                "night_multiplier": NIGHT_MULT[city]
            })
    fare_df = pd.DataFrame(fare_rows)

    np.random.seed(42)
    merged_df["vehicle_type"] = np.random.choice(
        ["Bike","Auto","Mini","Sedan","SUV"], len(merged_df)
    )
    merged_df = merged_df.merge(fare_df, on=["city","vehicle_type"], how="left")

    def gen_dist(n):
        d=[]
        for _ in range(n):
            p=np.random.rand()
            if p<0.7: d.append(np.random.uniform(2,8))
            elif p<0.9: d.append(np.random.uniform(8,15))
            else: d.append(np.random.uniform(15,35))
        return d

    merged_df["distance_km"] = np.array(gen_dist(len(merged_df))).round(2)
    expanded_df = merged_df.loc[merged_df.index.repeat(10)].reset_index(drop=True)

    def gen_demand(row):
        h=row["hour"]; r=row["rain"]
        d = 4 if (7<=h<=10 or 17<=h<=21) else 2
        if r>0: d+=1
        return min(d,5)

    def gen_surge(row):
        d=row["demand_level"]; r=row["rain"]; t=row["congestion_level"]
        s = {1:1.0,2:1.0,3:1.0,4:1.15,5:1.3}.get(d,1.0)
        if r>1: s+=0.05
        if t>40: s+=0.05
        return round(min(max(s,1.0),2.0),2)

    expanded_df["demand_level"]    = expanded_df.apply(gen_demand, axis=1)
    expanded_df["surge_multiplier"] = expanded_df.apply(gen_surge, axis=1)

    no_surge_idx = expanded_df.sample(frac=0.30, random_state=42).index
    expanded_df["surge_flag"] = 1
    expanded_df.loc[no_surge_idx, "surge_flag"] = 0

    def compute_fare(r):
        fare = r["base_fare_per_km"] * r["distance_km"]
        fare = max(fare, r["minimum_fare"])
        fare += r["booking_fee"]
        fare *= r["night_multiplier"]
        fare *= r["surge_multiplier"]
        return round(fare,2)

    expanded_df["total_fare"] = expanded_df.apply(compute_fare, axis=1)

    # Encode
    le_city    = LabelEncoder().fit(expanded_df["city"])
    le_vehicle = LabelEncoder().fit(expanded_df["vehicle_type"])
    le_season  = LabelEncoder().fit(expanded_df["season"])

    expanded_df["city"]         = le_city.transform(expanded_df["city"])
    expanded_df["vehicle_type"] = le_vehicle.transform(expanded_df["vehicle_type"])
    expanded_df["season"]       = le_season.transform(expanded_df["season"])

    Xc = expanded_df[[
        "distance_km","hour","month","temperature_2m",
        "relativehumidity_2m","rain","congestion_level",
        "vehicle_type","season","demand_level"
    ]]
    yc = expanded_df["surge_flag"]

    Xr = expanded_df[[
        "distance_km","base_fare_per_km","minimum_fare","booking_fee",
        "temperature_2m","relativehumidity_2m","rain","demand_level",
        "congestion_level","vehicle_type","night_multiplier"
    ]]
    yr = expanded_df["surge_multiplier"]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for tr, te in sss.split(Xc, yc):
        Xc_train, Xc_test = Xc.iloc[tr], Xc.iloc[te]
        yc_train, yc_test = yc.iloc[tr], yc.iloc[te]

    Xr_train,Xr_test,yr_train,yr_test = train_test_split(Xr,yr,test_size=0.2,random_state=42)

    # Train best classifier — LightGBM
    clf = LGBMClassifier(verbose=-1)
    clf.fit(Xc_train, yc_train)
    cls_preds = clf.predict(Xc_test)
    cls_acc   = accuracy_score(yc_test, cls_preds)

    # Train best regressor — Extra Trees
    reg = ExtraTreesRegressor(n_estimators=200, random_state=42)
    reg.fit(Xr_train, yr_train)
    reg_preds = reg.predict(Xr_test)
    reg_r2    = r2_score(yr_test, reg_preds)
    reg_mae   = mean_absolute_error(yr_test, reg_preds)
    reg_rmse  = np.sqrt(mean_squared_error(yr_test, reg_preds))

    # Save models
    joblib.dump(clf, MODEL_PATH_CLS)
    joblib.dump(reg, MODEL_PATH_REG)
    joblib.dump({
        "le_city": le_city,
        "le_vehicle": le_vehicle,
        "le_season": le_season
    }, ENCODER_PATH)

    metrics = {
        "cls_acc": cls_acc,
        "reg_r2": reg_r2,
        "reg_mae": reg_mae,
        "reg_rmse": reg_rmse
    }

    return clf, reg, le_city, le_vehicle, le_season, metrics


# ─── PREDICTION ───────────────────────────────────────────────────────────────
def predict(clf, reg, le_vehicle, le_season,
            city, vehicle_type, distance_km, hour, month,
            temperature, humidity, rain, demand_level):

    congestion = TRAFFIC[city]
    vtype_enc  = le_vehicle.transform([vehicle_type])[0]
    season_enc = le_season.transform([to_season(month)])[0]

    Xc_input = pd.DataFrame([{
        "distance_km": distance_km,
        "hour": hour,
        "month": month,
        "temperature_2m": temperature,
        "relativehumidity_2m": humidity,
        "rain": rain,
        "congestion_level": congestion,
        "vehicle_type": vtype_enc,
        "season": season_enc,
        "demand_level": demand_level
    }])

    surge_flag = clf.predict(Xc_input)[0]
    surge_prob = clf.predict_proba(Xc_input)[0][1]

    base_per_km  = FARE_DATA[city][vehicle_type]
    min_fare     = MIN_FARE[vehicle_type]
    booking      = BOOKING_FEE[vehicle_type]
    night_mult   = NIGHT_MULT[city]

    Xr_input = pd.DataFrame([{
        "distance_km": distance_km,
        "base_fare_per_km": base_per_km,
        "minimum_fare": min_fare,
        "booking_fee": booking,
        "temperature_2m": temperature,
        "relativehumidity_2m": humidity,
        "rain": rain,
        "demand_level": demand_level,
        "congestion_level": congestion,
        "vehicle_type": vtype_enc,
        "night_multiplier": night_mult
    }])

    surge_mult = reg.predict(Xr_input)[0]

    base_fare = max(base_per_km * distance_km, min_fare)
    fare_before_surge = (base_fare + booking) * night_mult
    total_fare = fare_before_surge * surge_mult

    return {
        "surge_flag":  int(surge_flag),
        "surge_prob":  round(surge_prob * 100, 1),
        "surge_mult":  round(surge_mult, 2),
        "base_fare":   round(base_fare, 2),
        "booking_fee": booking,
        "night_mult":  night_mult,
        "fare_before_surge": round(fare_before_surge, 2),
        "total_fare":  round(total_fare, 2),
        "congestion":  congestion
    }


# ─── LIVE WEATHER FETCH (single city) ─────────────────────────────────────────
def get_current_weather(city):
    lat, lon = CITIES[city]
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current_weather": True,
            "hourly": "relativehumidity_2m,rain",
            "timezone": "Asia/Kolkata",
            "forecast_days": 1
        }
        data = requests.get(url, params=params, timeout=8).json()
        cw   = data["current_weather"]
        hour = int(cw["time"].split("T")[1].split(":")[0])
        hum  = data["hourly"]["relativehumidity_2m"][hour]
        rain = data["hourly"]["rain"][hour]
        return round(cw["temperature"], 1), hum, round(rain, 2), True
    except:
        return 28.0, 65, 0.0, False


# ─── MAIN APP ─────────────────────────────────────────────────────────────────
def main():

    # Header
    st.markdown("""
    <div class="header-block">
        <p class="header-sub">🚕 AI-Powered · Live Weather · Multi-City</p>
        <h1 class="header-title">Surge Price Predictor</h1>
        <p style="color:#7070a0;font-size:0.88rem;margin:0;font-family:'Space Mono',monospace;">
            LightGBM + ExtraTrees · Real-time Open-Meteo API · 7 Indian Cities
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Train models (cached)
    with st.spinner("🔧 Training models on live weather data..."):
        clf, reg, le_city, le_vehicle, le_season, metrics = build_and_train()

    # Model metrics bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Classifier Accuracy</div>
            <div class="metric-value" style="color:#f5a623">{metrics['cls_acc']*100:.1f}%</div>
            <div class="metric-label">LightGBM</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Regressor R²</div>
            <div class="metric-value" style="color:#39d353">{metrics['reg_r2']:.4f}</div>
            <div class="metric-label">Extra Trees</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value" style="color:#e84855">{metrics['reg_mae']:.4f}</div>
            <div class="metric-label">Surge Multiplier</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value" style="color:#a78bfa">{metrics['reg_rmse']:.4f}</div>
            <div class="metric-label">Surge Multiplier</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SIDEBAR INPUTS ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 0 8px 0;">
            <p style="font-family:'Space Mono',monospace;font-size:0.68rem;
               color:#7070a0;text-transform:uppercase;letter-spacing:2px;margin:0;">
                Configure Ride
            </p>
            <h2 style="margin:4px 0 0 0;font-size:1.3rem;font-weight:800;">
                Ride Parameters
            </h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">📍 Location</div>', unsafe_allow_html=True)
        city = st.selectbox("City", list(CITIES.keys()), label_visibility="collapsed")

        st.markdown('<div class="section-label">🚗 Vehicle</div>', unsafe_allow_html=True)
        vehicle_type = st.selectbox(
            "Vehicle Type", ["Bike","Auto","Mini","Sedan","SUV"],
            label_visibility="collapsed"
        )

        st.markdown('<div class="section-label">📏 Trip</div>', unsafe_allow_html=True)
        distance_km = st.slider("Distance (km)", 1.0, 35.0, 6.0, 0.5)

        st.markdown('<div class="section-label">🕐 Time</div>', unsafe_allow_html=True)
        col_h, col_m = st.columns(2)
        with col_h:
            hour = st.number_input("Hour (0–23)", 0, 23, 8)
        with col_m:
            month = st.number_input("Month (1–12)", 1, 12, 6)

        st.markdown('<div class="section-label">🌤 Weather</div>', unsafe_allow_html=True)

        use_live = st.toggle("Use Live Weather", value=True)

        if use_live:
            temp, humidity, rain, ok = get_current_weather(city)
            if ok:
                st.success(f"Live data: {temp}°C, {humidity}% RH, {rain}mm rain")
            else:
                st.warning("Couldn't fetch live data. Using defaults.")
                temp, humidity, rain = 28.0, 65, 0.0
        else:
            temp     = st.slider("Temperature (°C)", 10.0, 45.0, 28.0, 0.5)
            humidity = st.slider("Humidity (%)", 20, 100, 65)
            rain     = st.slider("Rain (mm)", 0.0, 20.0, 0.0, 0.5)

        st.markdown('<div class="section-label">📊 Demand</div>', unsafe_allow_html=True)
        demand_level = st.select_slider(
            "Demand Level", options=[1,2,3,4,5], value=3,
            label_visibility="collapsed"
        )
        demand_labels = {1:"Very Low",2:"Low",3:"Medium",4:"High",5:"Very High"}
        st.caption(f"Current: **{demand_labels[demand_level]}**")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Predict Surge", use_container_width=True)

    # ── RIGHT PANEL ────────────────────────────────────────────────────────────
    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        if predict_btn:
            result = predict(
                clf, reg, le_vehicle, le_season,
                city, vehicle_type, distance_km, hour, month,
                temp, humidity, rain, demand_level
            )

            # Surge classification result
            if result["surge_flag"] == 1:
                st.markdown(f"""
                <div class="result-surge">
                    <div class="result-icon">⚠️</div>
                    <p class="result-title" style="color:#e84855;">SURGE ACTIVE</p>
                    <p class="result-detail">
                        {result['surge_prob']}% surge probability · {result['surge_mult']}× multiplier
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-no-surge">
                    <div class="result-icon">✅</div>
                    <p class="result-title" style="color:#39d353;">NO SURGE</p>
                    <p class="result-detail">
                        {result['surge_prob']}% surge probability · Normal pricing
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Fare breakdown
            st.markdown(f"""
            <div class="fare-box">
                <p class="section-label" style="margin-top:0;">💰 Fare Breakdown</p>
                <div class="fare-row">
                    <span class="fare-label">Base Fare</span>
                    <span class="fare-value">₹{result['base_fare']}</span>
                </div>
                <div class="fare-row">
                    <span class="fare-label">Booking Fee</span>
                    <span class="fare-value">₹{result['booking_fee']}</span>
                </div>
                <div class="fare-row">
                    <span class="fare-label">Night Multiplier</span>
                    <span class="fare-value">{result['night_mult']}×</span>
                </div>
                <div class="fare-row">
                    <span class="fare-label">Surge Multiplier</span>
                    <span class="fare-value" style="color:#f5a623">{result['surge_mult']}×</span>
                </div>
                <div class="fare-row">
                    <span class="fare-label">Total Estimated Fare</span>
                    <span class="fare-value" style="color:#f5a623;font-size:1.2rem">₹{result['total_fare']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Context chips
            st.markdown("<br>", unsafe_allow_html=True)
            peak = "🔴 Peak Hours" if (7<=hour<=10 or 17<=hour<=21) else "🟢 Off-Peak"
            rain_tag = "🌧 Rainy" if rain>0 else "☀️ Clear"
            traffic_tag = f"🚦 Traffic: {result['congestion']}%"
            st.markdown(f"""
            <div style="display:flex;gap:8px;flex-wrap:wrap;">
                <span style="background:#1a1a26;border:1px solid #2a2a3d;
                      border-radius:20px;padding:4px 14px;
                      font-family:'Space Mono',monospace;font-size:0.72rem;">{peak}</span>
                <span style="background:#1a1a26;border:1px solid #2a2a3d;
                      border-radius:20px;padding:4px 14px;
                      font-family:'Space Mono',monospace;font-size:0.72rem;">{rain_tag}</span>
                <span style="background:#1a1a26;border:1px solid #2a2a3d;
                      border-radius:20px;padding:4px 14px;
                      font-family:'Space Mono',monospace;font-size:0.72rem;">{traffic_tag}</span>
                <span style="background:#1a1a26;border:1px solid #2a2a3d;
                      border-radius:20px;padding:4px 14px;
                      font-family:'Space Mono',monospace;font-size:0.72rem;">{to_season(month)}</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#12121a;border:1px dashed #2a2a3d;border-radius:16px;
                 padding:60px 40px;text-align:center;margin-top:8px;">
                <p style="font-size:3rem;margin:0">⚡</p>
                <p style="font-family:'Space Mono',monospace;font-size:0.8rem;
                   color:#7070a0;margin:12px 0 0 0;letter-spacing:1px;">
                   SET PARAMETERS → CLICK PREDICT
                </p>
            </div>
            """, unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-label">📊 City Comparison</div>', unsafe_allow_html=True)

        # Quick table — all cities at current settings
        if predict_btn:
            rows = []
            for c in CITIES:
                t, h, r, _ = get_current_weather(c) if use_live else (temp, humidity, rain, True)
                cong = TRAFFIC[c]
                d_lvl = demand_level
                bpk = FARE_DATA[c][vehicle_type]
                mf  = MIN_FARE[vehicle_type]
                bf  = BOOKING_FEE[vehicle_type]
                nm  = NIGHT_MULT[c]

                vtype_enc  = le_vehicle.transform([vehicle_type])[0]
                season_enc = le_season.transform([to_season(month)])[0]

                Xc_row = pd.DataFrame([{
                    "distance_km": distance_km, "hour": hour, "month": month,
                    "temperature_2m": t, "relativehumidity_2m": h, "rain": r,
                    "congestion_level": cong, "vehicle_type": vtype_enc,
                    "season": season_enc, "demand_level": d_lvl
                }])
                sflag = clf.predict(Xc_row)[0]

                Xr_row = pd.DataFrame([{
                    "distance_km": distance_km, "base_fare_per_km": bpk,
                    "minimum_fare": mf, "booking_fee": bf,
                    "temperature_2m": t, "relativehumidity_2m": h,
                    "rain": r, "demand_level": d_lvl, "congestion_level": cong,
                    "vehicle_type": vtype_enc, "night_multiplier": nm
                }])
                smult = reg.predict(Xr_row)[0]

                base = max(bpk * distance_km, mf)
                total = (base + bf) * nm * smult

                rows.append({
                    "City": c,
                    "Surge": "⚠️ Yes" if sflag else "✅ No",
                    "Multiplier": f"{smult:.2f}×",
                    "Est. Fare (₹)": f"₹{total:.0f}",
                    "Traffic": f"{TRAFFIC[c]}%"
                })

            df_compare = pd.DataFrame(rows)
            st.dataframe(
                df_compare,
                use_container_width=True,
                hide_index=True
            )

        st.markdown('<div class="section-label" style="margin-top:24px;">🛠 Model Info</div>',
                    unsafe_allow_html=True)

        with st.expander("Classification — LightGBM"):
            st.markdown(f"""
            **Task:** Surge Yes/No (Binary)  
            **Accuracy:** `{metrics['cls_acc']*100:.2f}%`  
            **Features:** distance, hour, month, temperature, humidity, rain, congestion, vehicle, season, demand  
            **Split:** 80/20 Stratified
            """)

        with st.expander("Regression — Extra Trees"):
            st.markdown(f"""
            **Task:** Surge Multiplier (1.0 – 2.0)  
            **R²:** `{metrics['reg_r2']:.4f}`  
            **MAE:** `{metrics['reg_mae']:.4f}`  
            **RMSE:** `{metrics['reg_rmse']:.4f}`  
            **Features:** distance, fare params, weather, demand, congestion, vehicle, night multiplier  
            **Estimators:** 200 trees
            """)

        with st.expander("Data Pipeline"):
            st.markdown("""
            **Source:** Open-Meteo API (live hourly weather)  
            **Cities:** Bangalore, Chennai, Mumbai, Delhi, Hyderabad, Pune, Kolkata  
            **Expansion:** 10× repeat for training volume  
            **Demand Logic:** Hour-based + rain boost  
            **Surge Logic:** Demand + rain + congestion thresholds  
            **Encoding:** LabelEncoder for city, vehicle, season
            """)


if __name__ == "__main__":
    main()
