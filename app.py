import streamlit as st
import pandas as pd
import joblib

# Load model yang sudah dilatih
iso = joblib.load("model_iso.pkl")
rf = joblib.load("model_rf.pkl")

# Definisi fitur sesuai training
feature_names = [
    "Type", 
    "Air temperature [K]", 
    "Process temperature [K]", 
    "Rotational speed [rpm]", 
    "Torque [Nm]", 
    "Tool wear [min]"
]

# Mapping label output
label_map = {
    0: "✅ Normal",
    1: "❌ Tool Wear Failure (TWF)",
    2: "❌ Heat Dissipation Failure (HDF)",
    3: "❌ Power Failure (PWF)",
    4: "❌ Overstrain Failure (OSF)",
    5: "❌ Random Failure (RNF)"
}

st.title("🚀 Predictive Maintenance App")
st.markdown("Masukkan data sensor mesin untuk deteksi **Normal**, **Failure**, atau **Unknown/Anomaly**.")

# Input manual user
inputs = {}
for col in feature_names:
    if col == "Type":
        inputs[col] = st.selectbox("Product Type", ["L", "M", "H"])
    else:
        inputs[col] = st.number_input(col, value=0.0)

# Convert ke DataFrame
input_df = pd.DataFrame([inputs])

# Encode Type
input_df["Type"] = input_df["Type"].map({"L": 0, "M": 1, "H": 2})

# Prediksi
if st.button("Predict"):
    anomaly_flag = iso.predict(input_df)[0]
    if anomaly_flag == -1:
        st.error("⚠️ Unknown / Anomaly Detected (perlu investigasi manual)")
    else:
        pred = rf.predict(input_df)[0]
        st.success(f"Hasil Prediksi: {label_map[pred]}")
