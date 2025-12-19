import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Vignesh-vigu/Engine-Predictive-Maintenance-MLOps", filename="epm_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Engine Predictive Maintenance
st.title("🔧 Engine Predictive Maintenance App")
st.write("""
This application predicts whether an **engine requires maintenance**
based on real-time sensor parameters.
""")

# ------------------------------
# Input fields (DATA-DRIVEN FROM CSV STATS)
# ------------------------------
engine_rpm = st.number_input(
    "Engine rpm",
    min_value=61.0,
    max_value=2239.0,
    value=800.0,
    step=10.0
)

lub_oil_pressure = st.number_input(
    "Lub oil pressure (bar)",
    min_value=0.0,
    max_value=7.3,
    value=3.3,
    step=0.1
)

fuel_pressure = st.number_input(
    "Fuel pressure (bar)",
    min_value=0.0,
    max_value=21.2,
    value=6.6,
    step=0.1
)

coolant_pressure = st.number_input(
    "Coolant pressure (bar)",
    min_value=0.0,
    max_value=7.5,
    value=2.3,
    step=0.1
)

lub_oil_temp = st.number_input(
    "lub oil temp (°C)",
    min_value=71.0,
    max_value=90.0,
    value=77.0,
    step=0.5
)

coolant_temp = st.number_input(
    "Coolant temp (°C)",
    min_value=61.0,
    max_value=196.0,
    value=78.0,
    step=0.5
)

# ------------------------------
# Prepare Input Data
# ------------------------------
input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp
}])


# ------------------------------
# Predict Button
# ------------------------------
if st.button("Predict Maintenance Requirement"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.error("⚠️ Maintenance Required")
    else:
        st.success("✅ Engine Operating Normally")
