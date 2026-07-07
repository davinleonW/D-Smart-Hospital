import streamlit as stt
import pandas as pd
import pickle

with open("Davin-hospital_model.pkl","rb")as f
bundle = pickle.load(f)
st.write("Connected")
