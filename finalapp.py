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
        # Map categorical variables
        si_mapping = {'Minor Damage': 0, 'Total Loss': 1, 'Trivial Damage': 2}
        si = si_mapping[si]
        oc_mapping = {'Armed Forces': 0, 'Craft RepairExec Managerial': 1, 'Farming Fishing': 2, 'Handlers Cleaners': 3,
              'Machine Op Inspct': 4, 'Other Service': 5, 'Priv House Serv': 6, 'Prof Specialty': 7,
              'Protective Serv': 8, 'Sales': 9, 'Tech Support': 10, 'Transport Moving': 11}
        oc = oc_mapping[oc]
        gr1_mapping = {'Parked Car': 0, 'Single Vehicle Collision': 1, 'Vehicle Theft Incident': 2}
        gr1 = gr1_mapping[gr1]
        ct_mapping = {'Front Collision': 0, 'Rear Collision': 1, 'Side Collision': 2}
        ct = ct_mapping[ct]
        cp_mapping = {'Fire Authority': 0, 'No Contact': 1, 'Other Contacted': 2, 'Contacted Police': 3}
        cp = cp_mapping[cp]
        pp_mapping = {'Low Premium': 0, 'Medium Premium': 1, 'Very High Premium': 2, 'Very Low Premium': 3}
        pp = pp_mapping[pp]


    input_data = np.array([[ag, wt, ca, si, si1, ih, oc, gr1, ct, cp, pp, cg, cl]])
    st.write("Input Data for Prediction:", input_data)
    st.text(result[0])

st.markdown("Developed by WBL Intern Khan Sana  at NIELIT Daman")
