import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
clf = pickle.load(open("dtc_model.pkl","rb"))

def predict(data):
    clf = pickle.load(open("dtc_model.pkl","rb"))
    return clf.predict(data)

st.title("Insurance Fraud Detection Project using Machine Learning")
st.markdown("This Model detects fraud")

st.header("Parameters to Detect Fraud")
col1, col2, col3 = st.columns(3)

with col1:
    # Add sliders for numerical inputs
    capital_gains = st.slider("Capital Gains", 0, 100000, step=1000)
    capital_loss = st.slider("Capital Loss", -100000, 0, step=1000)
    incident_hour = st.slider("Incident Hour of the Day", 0, 24, step=1)
    vehicles_involved = st.slider("Number of Vehicles Involved", 1, 10, step=1)
    witnesses = st.slider("Number of Witnesses", 0, 10, step=1)
    claim_amount = st.slider("Total Claim Amount", 5000, 100000, step=500)

with col2:
    sex = st.selectbox("Sex", ["FEMALE", "MALE"])
    occupation = st.selectbox(
        "Occupation",
        ["adm-clerical", "armed-forces", "craft-repair", "exec-managerial", 
         "farming-fishing", "handlers-cleaners", "machine-op-inspct", 
         "other-service", "priv-house-serv", "prof-specialty", 
         "protective-serv", "sales", "tech-support", "transport-moving"]
    )

with col3:
    incident_type = st.selectbox(
        "Incident Type",
        ["Multi-vehicle Collision", "Parked Car", "Single Vehicle Collision", "Vehicle Theft"]
    )
    collision_type = st.selectbox(
        "Collision Type",
        ["?", "Front Collision", "Rear Collision", "Side Collision"]
    )
    incident_severity = st.selectbox(
        "Incident Severity",
        ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"]
    )
   
    # Construct the input data as a numerical array
    input_data = np.array([[
    capital_gains, capital_loss, incident_hour, vehicles_involved, witnesses, claim_amount,
    # Add encoded categorical variables here
    insured_sex_FEMALE, insured_sex_MALE,  # Example for sex
    *occupation_columns.values(),         # Unpack one-hot encoded occupation
    # Include all other one-hot encoded features
]])
    result = predict(input_data)
    if result[0] == 1:
        st.success("Fraud Reported")
    else:
        st.info("No Fraud Reported")


st.markdown("Developed by WBL Intern Khan Sana  at NIELIT Daman")
