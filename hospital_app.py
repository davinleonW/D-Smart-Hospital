import streamlit as st
import pandas as pd
import pickle

# ===============================
# LOAD MACHINE LEARNING MODEL
# ===============================

with open("Davin-hospital_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]

features = bundle["features"]
cols_to_scale = bundle["cols_to_scale"]

dept_map_inv = bundle["dept_map_inv"]

gender_map = bundle["gender_map"]
temp_map = bundle["temp_map"]
hr_map = bundle["hr_map"]
dur_map = bundle["dur_map"]
cc_map = bundle["cc_map"]


# ===============================
# DEPARTMENT INFORMATION
# ===============================

DEPT_INFO = {
    "Respiratory Medicine": {
        "icon": "🫁",
        "desc": "Specialises in conditions affecting the lungs and airways.",
        "next": [
            "Visit Level 2, Wing B",
            "Estimated wait: 15–25 minutes",
            "Please wear a mask"
        ]
    },

    "Cardiology": {
        "icon": "❤️",
        "desc": "Specialises in heart and cardiovascular conditions.",
        "next": [
            "Visit Level 3, Wing A",
            "Estimated wait: 20–30 minutes",
            "Bring previous ECG reports"
        ]
    },

    "Gastroenterology": {
        "icon": "🫃",
        "desc": "Specialises in digestive system conditions.",
        "next": [
            "Visit Level 1, Wing C",
            "Estimated wait: 10–20 minutes"
        ]
    },

    "Neurology": {
        "icon": "🧠",
        "desc": "Specialises in brain and nervous system conditions.",
        "next": [
            "Visit Level 4, Wing A",
            "Bring current medications list"
        ]
    },

    "General Medicine": {
        "icon": "🩺",
        "desc": "General health consultation.",
        "next": [
            "Visit Level 1, Wing A"
        ]
    },

    "Dermatology": {
        "icon": "🔬",
        "desc": "Specialises in skin conditions.",
        "next": [
            "Visit Level 2, Wing D"
        ]
    }
}


# ===============================
# STREAMLIT TITLE
# ===============================

st.title("🏥 Smart Hospital Patient Navigator")

st.write("Fill in the patient's information below.")

# ===============================
# PATIENT INFORMATION
# ===============================

st.header("Patient Information")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

# ===============================
# SYMPTOMS
# ===============================

st.header("Symptoms")

col1, col2 = st.columns(2)

with col1:
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")
    headache = st.checkbox("Headache")
    chest_pain = st.checkbox("Chest Pain")
    stomach_pain = st.checkbox("Stomach Pain")

with col2:
    shortness_breath = st.checkbox("Shortness of Breath")
    nausea_vomiting = st.checkbox("Nausea / Vomiting")
    dizziness = st.checkbox("Dizziness")
    skin_rash = st.checkbox("Skin Rash")

# ===============================
# PATIENT CONDITION
# ===============================

st.header("Patient Condition")

temperature_level = st.selectbox(
    "Temperature",
    options=list(temp_map.keys())
)

heart_rate_level = st.selectbox(
    "Heart Rate",
    options=list(hr_map.keys())
)

duration = st.selectbox(
    "Duration of Symptoms",
    options=list(dur_map.keys())
)

chief_complaint = st.selectbox(
    "Chief Complaint",
    options=list(cc_map.keys())
)

# ===============================
# MEDICAL HISTORY
# ===============================

st.header("Medical History")

hypertension = st.checkbox("Hypertension")

heart_disease = st.checkbox("Heart Disease")

asthma = st.checkbox("Asthma")

# ===============================
# PREDICT BUTTON
# ===============================

predict_button = st.button("Predict Department")

# ===============================
# MAKE PREDICTION
# ===============================

if predict_button:

    # Create patient data
    patient = pd.DataFrame([{
        "age": age,
        "gender": gender_map[gender],

        "fever": int(fever),
        "cough": int(cough),
        "headache": int(headache),
        "chest_pain": int(chest_pain),
        "stomach_pain": int(stomach_pain),
        "shortness_breath": int(shortness_breath),
        "nausea_vomiting": int(nausea_vomiting),
        "dizziness": int(dizziness),
        "skin_rash": int(skin_rash),

        "temperature_level": temp_map[temperature_level],
        "heart_rate_level": hr_map[heart_rate_level],
        "duration": dur_map[duration],

        "asthma": int(asthma),
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),

        "chief_complaint": cc_map[chief_complaint]
    }])

    # Scale numerical features
    patient_scaled = patient.copy()

    patient_scaled[cols_to_scale] = scaler.transform(
        patient[cols_to_scale]
    )

    # Predict department
    prediction = model.predict(
        patient_scaled[features]
    )[0]

    # Predict confidence
    probability = model.predict_proba(
        patient_scaled[features]
    )[0]

    department = dept_map_inv[prediction]

    confidence = probability[prediction] * 100

    # ===============================
    # SHOW RESULT
    # ===============================

    st.divider()

    st.header("Prediction Result")

    info = DEPT_INFO.get(department)

    if info:

        st.success(
            f"{info['icon']} Recommended Department: {department}"
        )

        st.write(f"**Confidence:** {confidence:.1f}%")

        st.write("### Description")
        st.write(info["desc"])

        st.write("### What should the patient do?")

        for step in info["next"]:
            st.write(f"✅ {step}")

    else:

        st.success(f"Recommended Department: {department}")

        st.write(f"Confidence: {confidence:.1f}%")

    st.warning(
        "This AI recommendation is only for educational purposes and is not a medical diagnosis."
    )
  


















