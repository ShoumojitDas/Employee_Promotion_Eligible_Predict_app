import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load dataset (for encoding reference)
data = pd.read_csv(r"C:\Users\shoum\Employee_Performance Project\Data\Employee_Performance_Dataset.csv")

# Drop unnecessary columns
data = data.drop(["Employee ID", "Name"], axis=1)

# Load trained model
model = joblib.load(r"C:\Users\shoum\Employee_Performance Project\Emp_Promotion_check app\model.pkl")

st.set_page_config(page_title="Employee Promotion Predictor")

st.title("Employee Promotion Prediction")

st.write(
"""
This app predicts whether an employee is eligible for promotion
based on performance metrics.
"""
)

st.sidebar.header("Enter Employee Details")

# User Inputs
department = st.sidebar.selectbox("Department", data["Department"].unique())
job_role = st.sidebar.selectbox("Job Role", data["Job Role"].unique())

performance_score = st.sidebar.slider("Performance Score", 0, 100, 50)
kpi_score = st.sidebar.slider("KPI Score", 0, 100, 50)

attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
peer_rating = st.sidebar.slider("Peer Rating", 1, 5, 3)

task_completion = st.sidebar.slider("Task Completion (%)", 0, 100, 70)

work_hours = st.sidebar.slider("Work Hours Logged", 0, 60, 40)

manager_feedback = st.sidebar.selectbox(
    "Manager Feedback",
    data["Manager Feedback"].unique()
)

training_hours = st.sidebar.slider("Training Hours", 0, 50, 10)

# Create dataframe from inputs
input_data = pd.DataFrame({
    "Department":[department],
    "Job Role":[job_role],
    "Performance Score":[performance_score],
    "KPI Score":[kpi_score],
    "Attendance (%)":[attendance],
    "Peer Rating":[peer_rating],
    "Task Completion (%)":[task_completion],
    "Work Hours Logged":[work_hours],
    "Manager Feedback":[manager_feedback],
    "Training Hours":[training_hours]
})

# Apply Label Encoding
categorical_cols = ["Department","Job Role","Manager Feedback"]

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    input_data[col] = le.transform(input_data[col])

# Prediction
if st.button("Predict Promotion"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Employee is Eligible for Promotion")
    else:
        st.error("Employee is NOT Eligible for Promotion")