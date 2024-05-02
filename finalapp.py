import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
clf = pickle.load(open("rf_model_v1.pkl","rb"))

def predict(data):
    clf = pickle.load(open("rf_model_v1.pkl","rb"))
    return clf.predict(data)

st.title("Insurance Fraud Detection Project using Machine Learning")
st.markdown("This Model detects fraud")

st.header("Parameters to Detect Fraud")
col1,col2,col3 = st.columns(3)

with col1:
    ag = st.slider("Age", 18, 80, 2)
    wt = st.slider("Number of Witnesses", 1, 10, 2)
    ca = st.slider("Claim Amount", 5000.0, 100000.0, 0.5)
    si = st.selectbox("Incident Severity", ['Minor Damage', 'Total Loss', 'Trivial Damage'])
    si1 = st.slider("Number of Vehicles Involved", 1, 40, 2)
    ih = st.slider("Incident Hours of day", 1, 24, 2)

with col2:
    oc = st.selectbox("Occupation", ['Armed Forces', 'Craft RepairExec Managerial', 'Farming Fishing', 'Handlers Cleaners', 'Machine Op Inspct', 'Other Service', 'Priv House Serv', 'Prof Specialty', 'Protective Serv', 'Sales', 'Tech Support', 'Transport Moving'])
    gr1 = st.selectbox("Incident Type", ['Parked Car', 'Single Vehicle Collision', 'Vehicle Theft Incident'])
    ct = st.selectbox("Collision Type", ['Front Collision', 'Rear Collision', 'Side Collision'])

with col3:
    cp = st.selectbox("Authorities Contacted", ['Fire Authority', 'No Contact', 'Other Contacted', 'Contacted Police'])
    pp = st.selectbox("Policy Premium", ['Low Premium', 'Medium Premium', 'Very High Premium', 'Very Low Premium'])
    cg = st.slider("Capital Gains", 0.0, 100000.0, 0.0)
    cl = st.slider("Capital Loss", -100000.0, 100000.0, 0.0)

if st.button("Predict Fraud"):
    si = 1 if si in ['Minor Damage', 'Total Loss', 'Trivial Damage'] else 0
    oc = 1 if oc in ['Armed Forces', 'Craft RepairExec Managerial', 'Farming Fishing', 'Handlers Cleaners', 'Machine Op Inspct', 'Other Service', 'Priv House Serv', 'Prof Specialty', 'Protective Serv', 'Sales', 'Tech Support', 'Transport Moving'] else 0
    gr1 = 1 if gr1 in ['Parked Car', 'Single Vehicle Collision', 'Vehicle Theft Incident'] else 0
    ct = 1 if ct in ['Front Collision', 'Rear Collision', 'Side Collision'] else 0
    cp = 1 if cp in ['Fire Authority', 'No Contact', 'Other Contacted', 'Contacted Police'] else 0
    pp = 1 if pp in ['Low Premium', 'Medium Premium', 'Very High Premium', 'Very Low Premium'] else 0

    input_data = np.array([[ag,wt,ca,oc,gr1,ct,si,cp,pp,cg,si1,cl,ih,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    result = predict(input_data)
    st.text(result[0])

st.markdown("Developed by External Guide Avinash Pawar and WBL Intern Khan Sana at NIELIT Daman")
