import streamlit as st
import pandas as pd
import joblib

# ============================
# Load model yang sudah dilatih di Colab
# ============================
iso = joblib.load("model_iso.pkl")  # Isolation Forest
rf = joblib.load("model_rf.pkl")    # Random Forest

# Fitur yang dipakai saat training (harus sama persis!)
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

# ============================
# Streamlit UI
# ============================
st.title("🚀 Predictive Maintenance App")
st.markdown("Masukkan data sensor mesin untuk deteksi **Normal**, **Failure**, atau **Unknown/Anomaly**.")

# Input manual
inputs = {}
for col in feature_names:
    if col == "Type":
        inputs[col] = st.selectbox("Product Type", ["L", "M", "H"])
    else:
        inputs[col] = st.number_input(col, value=0.0)

# Konversi ke DataFrame dengan kolom urut sesuai feature_names
input_df = pd.DataFrame([inputs], columns=feature_names)

# Encode Type
input_df["Type"] = input_df["Type"].map({"L": 0, "M": 1, "H": 2})

# ============================
# Prediction Pipeline
# ============================
if st.button("Predict"):
    try:
        # Step 1: Anomaly Detection
        anomaly_flag = iso.predict(input_df)[0]

        if anomaly_flag == -1:
            st.error("⚠️ Unknown / Anomaly Detected (perlu investigasi manual)")
        else:
            # Step 2: Failure Classification
            pred = rf.predict(input_df)[0]
            st.success(f"Hasil Prediksi: {label_map[pred]}")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {str(e)}")
