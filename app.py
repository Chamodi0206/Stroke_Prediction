import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load(r"E:\Projects\Stroke_prediction\Models\RF_model.pkl")
scaler_normal = joblib.load(r"E:\Projects\Stroke_prediction\Models\robust_scaler_m.pkl")

# Feature group functions
def create_age_group(age):
    if age < 13: return "Children"
    elif 13 <= age < 18: return "Teen"
    elif 18 <= age < 45: return "Adult"
    elif 45 <= age < 60: return "MidAge"
    else: return "Elderly"

def create_glucose_group(glucose):
    if glucose < 70: return "Low"
    elif 70 <= glucose < 99: return "Normal"
    elif 99 <= glucose < 125: return "Pre-diabetes"
    elif 125 <= glucose < 180: return "Diabetes"
    else: return "High-risk"

def create_bmi_group(bmi):
    if bmi < 18.5: return "Under weight"
    elif 18.5 <= bmi < 25: return "Normal weight"
    elif 25 <= bmi < 30: return "Over weight"
    else: return "Obesity weight"

# Mappings
gender_mapping = {'Male': 1, 'Female': 0, 'Other': 2}
ever_married_mapping = {'Yes': 1, 'No': 0}
work_type_mapping = {'children': 0, 'Govt_job': 1, 'Never_worked': 2, 'Private': 3, 'Self-employed': 4}
residence_mapping = {'Urban': 1, 'Rural': 0}
smoking_status_mapping = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}
age_group_mapping = {'Children': 0, 'Teen': 1, 'Adult': 2, 'MidAge': 3, 'Elderly': 4}
glucose_group_mapping = {'Low': 0, 'Normal': 1, 'Pre-diabetes': 2, 'Diabetes': 3, 'High-risk': 4}
bmi_group_mapping = {'Under weight': 0, 'Normal weight': 1, 'Over weight': 2, 'Obesity weight': 3}

# Streamlit UI
st.title("🧠 Stroke Prediction App")

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 1, 100, 58)
hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["children", "Govt_job", "Never_worked", "Private", "Self-employed"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=40.0, max_value=300.0, value=149.75)
bmi = st.number_input("BMI (Leave blank for default)", value=27.0)
smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

if st.button("🔍 Predict Stroke Risk"):
    # Default BMI if NaN
    if pd.isna(bmi):
        bmi = 28.9

    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # Feature Engineering
    input_data['age_group'] = input_data['age'].apply(create_age_group)
    input_data['glucose_group'] = input_data['avg_glucose_level'].apply(create_glucose_group)
    input_data['bmi_group'] = input_data['bmi'].apply(create_bmi_group)

    # Encode
    input_data['gender'] = input_data['gender'].map(gender_mapping)
    input_data['ever_married'] = input_data['ever_married'].map(ever_married_mapping)
    input_data['work_type'] = input_data['work_type'].map(work_type_mapping)
    input_data['Residence_type'] = input_data['Residence_type'].map(residence_mapping)
    input_data['smoking_status'] = input_data['smoking_status'].map(smoking_status_mapping)
    input_data['age_group'] = input_data['age_group'].map(age_group_mapping)
    input_data['glucose_group'] = input_data['glucose_group'].map(glucose_group_mapping)
    input_data['bmi_group'] = input_data['bmi_group'].map(bmi_group_mapping)

    expected_columns = [
        'gender', 'age', 'hypertension', 'heart_disease',
        'ever_married', 'work_type', 'Residence_type',
        'avg_glucose_level', 'bmi', 'smoking_status',
        'age_group', 'glucose_group', 'bmi_group'
    ]
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[expected_columns]

    # Scale
    input_scaled_array = scaler_normal.transform(input_data)
    input_scaled = pd.DataFrame(input_scaled_array, columns=expected_columns)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        if prediction_proba[0][1] >= 0.7:
            st.error(f"🛑 High Risk of Stroke")
        else:
            st.warning(f"⚠️ Moderate Risk of Stroke ({prediction_proba[0][1]:.2f})")
    else:
        if prediction_proba[0][0] >= 0.7:
            st.success(f"✅ Not at Risk of Stroke")
        else:
            st.info(f"🔍 Slight Risk of Stroke ({prediction_proba[0][1]:.2f})")
