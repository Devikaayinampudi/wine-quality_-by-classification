import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🍷 Wine Quality Prediction",
    page_icon="🍇",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Background */
.stApp {
    background-color: white;
}

/* Main container */
.main {
    background-color: #87CEFA;
    padding: 25px;
    border-radius: 20px;
}

/* Input boxes */
input {
    background-color: #FFC0CB !important;
}

/* Prediction text */
.prediction {
    color: #FF1493;
    font-size: 26px;
    font-weight: bold;
}

/* Sidebar grapes */
.grapes {
    font-size: 70px;
    text-align: center;
    line-height: 1.2;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("winequality-red.csv")

X = df.drop("quality", axis=1)
y = df["quality"]

# Convert to classification labels
y = y.apply(lambda x: 0 if x <= 5 else (1 if x == 6 else 2))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("<div class='grapes'>🍇<br>🍇<br>🍇<br>🍇</div>", unsafe_allow_html=True)
st.sidebar.title("🍷 Wine App")
st.sidebar.info("Predict Wine Quality using Machine Learning")

# ---------------- MAIN UI ----------------
st.title("🍷 Wine Quality Prediction Dashboard")

st.markdown("### Enter Wine Chemical Properties")

col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
    citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)
    residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 2.5)
    chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.08)

with col2:
    volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
    free_sulfur = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 15.0)
    total_sulfur = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 46.0)
    density = st.number_input("Density", 0.990, 1.005, 0.997)

with col3:
    pH = st.number_input("pH", 2.5, 4.5, 3.3)
    sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.65)
    alcohol = st.number_input("Alcohol", 8.0, 15.0, 10.0)

# ---------------- PREDICTION ----------------
if st.button("🍷 Predict Wine Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                             residual_sugar, chlorides, free_sulfur,
                             total_sulfur, density, pH, sulphates, alcohol]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.markdown("<div class='prediction'>❌ Bad Quality Wine 🍷</div>", unsafe_allow_html=True)
    elif prediction == 1:
        st.markdown("<div class='prediction'>✅ Good Quality Wine 🍷</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction'>⭐ Excellent Quality Wine 🍷</div>", unsafe_allow_html=True)
