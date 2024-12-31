import streamlit as st
import numpy as np
import pickle

# Load the model globally
clf = pickle.load(open("dtc_model.pkl", "rb"))

def predict(data):
    """Predict fraud using the trained model."""
    return clf.predict(data)

st.title("Insurance Fraud Detection Project using Machine Learning")
st.markdown("This Model detects fraud")

st.header("Parameters to Detect Fraud")
col1, col2, col3 = st.columns(3)

# Collect numerical inputs
with col1:
    capital_gains = st.slider("Capital Gains", 0, 100000, step=1000)
    capital_loss = st.slider("Capital Loss", -100000, 0, step=1000)
    incident_hour = st.slider("Incident Hour of the Day", 0, 24, step=1)
    vehicles_involved = st.slider("Number of Vehicles Involved", 1, 10, step=1)
    witnesses = st.slider("Number of Witnesses", 0, 10, step=1)
    claim_amount = st.slider("Total Claim Amount", 5000, 100000, step=500)

# Collect categorical inputs
with col2:
    sex = st.selectbox("Sex", ["FEMALE", "MALE"])
    occupation = st.selectbox(
        "Occupation",
        ["adm-clerical", "armed-forces", "craft-repair", "exec-managerial", 
         "farming-fishing", "handlers-cleaners", "machine-op-inspct", 
         "other-service", "priv-house-serv", "prof-specialty", 
         "protective-serv", "sales", "tech-support", "transport-moving"]
    )
    hobby = st.selectbox("Hobby", ["chess", "cross-fit", "other"])

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
    authorities_contacted = st.selectbox(
        "Authorities Contacted",
        ["Ambulance", "Fire", "None", "Other", "Police"]
    )

# Encode categorical inputs
sex_encoded = [1, 0] if sex == "FEMALE" else [0, 1]
hobby_encoded = [1 if hobby == "chess" else 0, 1 if hobby == "cross-fit" else 0, 1 if hobby == "other" else 0]

occupation_columns = [
    "adm-clerical", "armed-forces", "craft-repair", "exec-managerial",
    "farming-fishing", "handlers-cleaners", "machine-op-inspct", 
    "other-service", "priv-house-serv", "prof-specialty", 
    "protective-serv", "sales", "tech-support", "transport-moving"
]
occupation_encoded = [1 if occupation == col else 0 for col in occupation_columns]

incident_type_columns = [
    "Multi-vehicle Collision", "Parked Car", "Single Vehicle Collision", "Vehicle Theft"
]
incident_type_encoded = [1 if incident_type == col else 0 for col in incident_type_columns]

collision_type_columns = ["?", "Front Collision", "Rear Collision", "Side Collision"]
collision_type_encoded = [1 if collision_type == col else 0 for col in collision_type_columns]

incident_severity_columns = [
    "Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"
]
incident_severity_encoded = [1 if incident_severity == col else 0 for col in incident_severity_columns]

authorities_contacted_columns = ["Ambulance", "Fire", "None", "Other", "Police"]
authorities_contacted_encoded = [1 if authorities_contacted == col else 0 for col in authorities_contacted_columns]

# Additional features
age_group_encoded = [0, 0, 0, 0, 0, 0, 0, 0]  # 8 age groups
customer_group_encoded = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 10 customer groups
policy_premium_group_encoded = [0, 0, 0, 0, 0]  # 5 policy premium groups

# Construct the input data array
input_data = np.array([[
    capital_gains, capital_loss, incident_hour, vehicles_involved, witnesses, claim_amount,
    *sex_encoded,
    *occupation_encoded,
    *hobby_encoded,
    *incident_type_encoded,
    *collision_type_encoded,
    *incident_severity_encoded,
    *authorities_contacted_encoded,
    *age_group_encoded,
    *customer_group_encoded,
    *policy_premium_group_encoded
]])

# Validate feature count
st.write("Input Data Shape:", input_data.shape)
if st.button("Predict Fraud"):
    if input_data.shape[1] != 67:
        st.error(f"Input data shape mismatch: {input_data.shape[1]} features, expected 67.")
    else:
        result = predict(input_data)
        if result[0] == 1:
            st.success("Fraud Reported")
        else:
            st.info("No Fraud Reported")
