import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️")

# ============================
# Load model & preprocessors
# ============================
@st.cache_resource
def load_artifacts():
    iso = joblib.load("model_iso.pkl")
    binary_best = joblib.load("model_binary_best.pkl")
    multi_best = joblib.load("model_multi_best.pkl")
    type_encoder = joblib.load("type_encoder.pkl")
    scaler_bin = joblib.load("scaler_bin.pkl")
    scaler_m = joblib.load("scaler_m.pkl")
    return iso, binary_best, multi_best, type_encoder, scaler_bin, scaler_m


iso, binary_best, multi_best, type_encoder, scaler_bin, scaler_m = load_artifacts()

# Fitur dasar (sebelum feature engineering)
BASE_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Fitur turunan (harus sama dengan di notebook)
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Temp_Diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Torque_x_Wear"] = df["Torque [Nm]"] * df["Tool wear [min]"]
    df["Speed_per_Torque"] = df["Rotational speed [rpm]"] / (df["Torque [Nm]"] + 1)
    return df


FINAL_FEATURES = BASE_FEATURES + ["Temp_Diff", "Torque_x_Wear", "Speed_per_Torque"]

# Mapping label output
label_map = {
    0: "✅ Normal",
    1: "❌ Tool Wear Failure (TWF)",
    2: "❌ Heat Dissipation Failure (HDF)",
    3: "❌ Power Failure (PWF)",
    4: "❌ Overstrain Failure (OSF)",
    5: "❌ Random Failure (RNF)",
}

# ============================
# UI
# ============================
st.title("⚙️ Predictive Maintenance App (LightGBM + Isolation Forest)")
st.markdown(
    """
    Pipeline prediksi:
    1. **Isolation Forest** → deteksi *anomaly*.
    2. **Binary LightGBM** → Normal vs Failure.
    3. **Multiclass LightGBM** → Jenis failure (TWF, HDF, dst).
    """
)

st.subheader("Input Data Sensor")

inputs = {}
inputs["Type"] = st.selectbox("Product Type", ["L", "M", "H"])
inputs["Air temperature [K]"] = st.number_input("Air temperature [K]", value=300.0)
inputs["Process temperature [K]"] = st.number_input("Process temperature [K]", value=310.0)
inputs["Rotational speed [rpm]"] = st.number_input("Rotational speed [rpm]", value=1500.0)
inputs["Torque [Nm]"] = st.number_input("Torque [Nm]", value=40.0)
inputs["Tool wear [min]"] = st.number_input("Tool wear [min]", value=100.0)

if st.button("🚀 Predict"):
    try:
        # 1. Buat DataFrame dari input
        df = pd.DataFrame([inputs], columns=BASE_FEATURES)

        # 2. Encode Type pakai LabelEncoder yang sama dengan training
        #    type_encoder.classes_ harus berisi ['H', 'L', 'M'] dari dataset asli
        df["Type"] = type_encoder.transform(df["Type"])

        # 3. Tambah fitur turunan
        df = add_engineered_features(df)

        # 4. Ambil hanya kolom FINAL_FEATURES (urutan harus sama)
        X = df[FINAL_FEATURES].copy()

        # 5. Scaling untuk binary & anomaly detection
        X_bin_scaled = scaler_bin.transform(X)

        # 6. Isolation Forest → cek anomaly
        iso_pred = iso.predict(X_bin_scaled)[0]  # 1 = normal, -1 = anomaly

        if iso_pred == -1:
            st.error("⚠️ Unknown / Anomaly Detected — data tidak sesuai pola mesin.")
        else:
            # 7. Binary prediction → Normal vs Failure
            is_failure = binary_best.predict(X_bin_scaled)[0]

            if is_failure == 0:
                st.success("Hasil: ✅ **Normal** (tidak terdeteksi failure).")
            else:
                st.warning("⚠️ Mesin terdeteksi mengalami **Failure**.")

                # 8. Scaling untuk multiclass
                X_multi_scaled = scaler_m.transform(X)

                # 9. Multiclass prediction → jenis failure
                fail_class = multi_best.predict(X_multi_scaled)[0]
                st.error(f"Jenis Failure: **{label_map.get(fail_class, 'Unknown')}**")

                # (Opsional) tampilkan probabilitas tiap kelas
                if hasattr(multi_best, "predict_proba"):
                    proba = multi_best.predict_proba(X_multi_scaled)[0]
                    prob_df = pd.DataFrame(
                        {
                            "Class": [label_map[i] for i in range(len(proba))],
                            "Probability": proba,
                        }
                    )
                    st.write("Probabilitas tiap jenis failure:")
                    st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}))

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {str(e)}")
