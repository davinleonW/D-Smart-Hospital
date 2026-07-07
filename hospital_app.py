import streamlit as st
import pandas as pd
import pickle

with open("Davin-hospital_model.pkl","rb")as f:
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

DEPT_INFO = {
  "Respiratory Medicine" : {
    "icon" : "🫁",
    "desc" : "Specialises in condition affecting the lungs and airways",
    "next" : [
      "Visit level 2, Wing B",
      "Estimated waiting : 15-25 Minutes"
    ]
  },
    "Cardiology" : {
    "icon" : "🫀",
    "desc" : "Specialises in heart and cardiovascular condition",
    "next" : [
      "Visit level 3, Wing A",
      "Estimated waiting : 20-30 Minutes"
      "Bring previous ECG reports"
    ]
  },
  "gastroenterology" : {
    "icon" : "🫃",
    "desc" : "Specialises in digestive system condition",
    "next" : [
      "Visit level 1, Wing C",
      "Estimated waiting : 10-20 Minutes"
    ]
  },
  "Neurology" : {
    "icon" : "🧠",
    "desc" : "Specialises in brain and nervous system",
    "next" : [
      "Visit level 4, Wing A",
      "Bring current medication list"
    ]
  },
    "General Medicine" : {
    "icon" : "💊",
    "desc" : "Specialises in General help consultation",
    "next" : [
      "Visit level 1, Wing A",
    ]
  },
  "Dermatology" : {
    "icon" : "🧴",
    "desc" : "Specialises in skin condition",
    "next" : [
      "Visit level 2, Wing B",
      "Bring current medication list"
    ]
  },
}
st.title("🏥 Smart Hospital Navigator")
st.write("Fill in The Patient Information Below")
st.header("Patient Information")

age = st.number_input(
  "Age",
  min_value=1,
  max_value=120,
  value=30 
)
gender = st.selectbox(
  "Gender",
  ["Female,Male"]
)




st.header("Symptoms")
col1,col2 = st.columns(2)

with col1:
  fever = st.checkbox("Fever🤒")
  cough = st.checkbox("Cough😮‍💨")
  headache = st.checkbox("Headache🤯")
  chest_pain = st.checkbox("Chest pain❤️‍🩹")
  stomach_pain = st.checkbox("Stomach pain🤢")

with col2 :
  shortness breath = st.checkbox("Shortnessbreath😮")
  nausea = st.checkbox("Nausea🤮")
  dizzy = st.checkbox("Dizzy😵‍💫")
  skin_rash = st.checkbox("Skin rash💥")




st.header("Patient Condition")

temperature_level = st.selectbox(
  "Temperature",
  options=list(temp_map.keys())
)
heart_rate_level = st.selectbox(
  "HeartRate",
  options=list(hr_map.keys())
)
duration_level = st.selectbox(
  "Duration symptoms",
  options=list(dur_map.keys())
)
chief_complaint_level = st.selectbox(
  "Chief Complaint",
  options=list(cc_map.keys())
)


st.header("Medical History")
hypertension= st.checkbox("Hypertension")
heart_disease= st.checkbox("Heart Disease")
asthma= st.checkbox("Asthma")


predict_button = st.button("Predict Department")

if predict_button:
  patient = pd.DataFrame([{
    "age" : age,
    "gender" : gender_map[gender},
    "fever" : int(fever),
    "chest_pain" : int(chest_pain),
    "stomach_pain" : int(stomach_pain),
    "shortness_breath" : int(shortness_breath),
    "dizziness" : int(dizziness),
    "skin_rash" : int(skin_rash),


    "temperature_level" : temp_map[temperature_level],
    "heart_rate_level" : hr_map[heart_rate_level],
    "duration" : dur_map[duration],
    
    "asthma" : int(asthma),
    "hypertension" : int(hypertension),
    "heart_disease" : int(heart_disease),

    
    "chief_complaint": cc_map[chief_complaint]
  }])
  
  patient_scaled = patient.copy()
  patient_scaled[cols_to_scale] = scaler.transform(
      patient[cols_to_scale]
  )
  
  prediction = model.predict(
    patient_scaled[features]
  )[0]

  probability = model.predict_proba(
    patient_scaled[features]
  )[0]
  
  department = dept_map_inv[prediction]
  
  confidence = probability[prediction] * 100
  
  st.divider()
  st.header("Prediction Result")
  
  if info:
    st.sucess(
      f"{info['icon']} Recomended Department: {department}"
    )
    st.write(f"**Confidence: ** {confidence:.1f}%")
    st.write("### Description")
    st.write(info["desc"])
    st.write("### What should the patient do?")
    for step in info ["next"]:
      st.write("✅ {step}")
   else :
     st.sucess(f"Recommended Department : {department}")
     st.write(f"Confidence: {confidence:.1f}%")
   st.warning("This AI Recommendation is for learning purposes only")
    
  


















