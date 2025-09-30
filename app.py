import streamlit as st
import pandas as pd
import joblib

# Load model
rf = joblib.load("model_rf.pkl")
iso = joblib.load("model_iso.pkl")

st.title("⚙️ Predictive Maintenance App")
st.write("Anomaly detection + Failure classification")

# Input form
type_map = {"L": 0, "M": 1, "H": 2}

type_val = st.selectbox("Product Type", ["L", "M", "H"])
air_temp = st.number_input("Air temperature [K]", value=300.0)
proc_temp = st.number_input("Process temperature [K]", value=310.0)
rot_speed = st.number_input("Rotational speed [rpm]", value=1500)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool wear [min]", value=100)

# Convert ke DataFrame
input_data = pd.DataFrame([{
    "Type": type_map[type_val],
    "Air temperature [K]": air_temp,
    "Process temperature [K]": proc_temp,
    "Rotational speed [rpm]": rot_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}])

if st.button("Predict"):
    # Anomaly Detection
    anomaly_flag = iso.predict(input_data)[0]
    anomaly_msg = "⚠️ Anomaly Detected" if anomaly_flag == -1 else "Normal Data"

    # Failure Classification
    pred = rf.predict(input_data)[0]
    label_map = {
        0: "Normal",
        1: "Tool Wear Failure (TWF)",
        2: "Heat Dissipation Failure (HDF)",
        3: "Power Failure (PWF)",
        4: "Overstrain Failure (OSF)",
        5: "Random Failure (RNF)"
    }
    result = label_map[pred]

    st.subheader("🔎 Hasil Prediksi")
    st.write("Anomaly Status:", anomaly_msg)
    st.write("Failure Type:", result)
