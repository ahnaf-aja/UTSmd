import streamlit as st
import pandas as pd
import pickle

# LOAD MODEL
clf_model = pickle.load(open("classification_model.pkl", "rb"))
reg_model = pickle.load(open("regression_model.pkl", "rb"))

st.set_page_config(page_title="Student Placement Predictor", layout="wide")

st.title("Student Placement & Salary Prediction")
st.write("Masukkan data mahasiswa untuk prediksi hasil placement dan estimasi gaji.")

# SIDEBAR INPUT
st.sidebar.header("Input Data Mahasiswa")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
branch = st.sidebar.selectbox("Branch", ["CS", "IT", "ECE", "ME", "CE"])
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)
tenth = st.sidebar.slider("10th %", 0.0, 100.0, 70.0)
twelfth = st.sidebar.slider("12th %", 0.0, 100.0, 70.0)
backlogs = st.sidebar.slider("Backlogs", 0, 10, 0)
study_hours = st.sidebar.slider("Study Hours/Day", 0, 12, 4)
attendance = st.sidebar.slider("Attendance %", 0.0, 100.0, 80.0)
projects = st.sidebar.slider("Projects Completed", 0, 10, 2)
internships = st.sidebar.slider("Internships", 0, 5, 1)
coding = st.sidebar.slider("Coding Skill (1-10)", 1, 10, 5)
communication = st.sidebar.slider("Communication Skill (1-10)", 1, 10, 5)
aptitude = st.sidebar.slider("Aptitude Skill (1-10)", 1, 10, 5)
hackathons = st.sidebar.slider("Hackathons", 0, 10, 1)
certifications = st.sidebar.slider("Certifications", 0, 10, 1)
sleep = st.sidebar.slider("Sleep Hours", 0, 12, 7)
stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
part_time = st.sidebar.selectbox("Part Time Job", ["Yes", "No"])
income = st.sidebar.selectbox("Family Income", ["Low", "Medium", "High"])
city = st.sidebar.selectbox("City Tier", ["Tier1", "Tier2", "Tier3"])
internet = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
extra = st.sidebar.selectbox("Extracurricular", ["Yes", "No"])

# DATAFRAME INPUT
input_data = pd.DataFrame([{
    'gender': gender,
    'branch': branch,
    'cgpa': cgpa,
    'tenth_percentage': tenth,
    'twelfth_percentage': twelfth,
    'backlogs': backlogs,
    'study_hours_per_day': study_hours,
    'attendance_percentage': attendance,
    'projects_completed': projects,
    'internships_completed': internships,
    'coding_skill_rating': coding,
    'communication_skill_rating': communication,
    'aptitude_skill_rating': aptitude,
    'hackathons_participated': hackathons,
    'certifications_count': certifications,
    'sleep_hours': sleep,
    'stress_level': stress,
    'part_time_job': part_time,
    'family_income_level': income,
    'city_tier': city,
    'internet_access': internet,
    'extracurricular_involvement': extra
}])

st.subheader("Input Data")
st.dataframe(input_data)

# FIX INPUT DATA

expected_cols = [
    'cgpa',
    'tenth_percentage',
    'twelfth_percentage',
    'backlogs',
    'study_hours_per_day',
    'attendance_percentage',
    'projects_completed',
    'internships_completed',
    'coding_skill_rating',
    'communication_skill_rating',
    'aptitude_skill_rating',
    'hackathons_participated',
    'certifications_count',
    'sleep_hours',
    'stress_level',
    'gender',
    'branch',
    'part_time_job',
    'family_income_level',
    'city_tier',
    'internet_access',
    'extracurricular_involvement'
]

for col in expected_cols:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[expected_cols]

# Fix dtype numeric
numeric_cols = [
    'cgpa',
    'tenth_percentage',
    'twelfth_percentage',
    'backlogs',
    'study_hours_per_day',
    'attendance_percentage',
    'projects_completed',
    'internships_completed',
    'coding_skill_rating',
    'communication_skill_rating',
    'aptitude_skill_rating',
    'hackathons_participated',
    'certifications_count',
    'sleep_hours',
    'stress_level'
]

input_data[numeric_cols] = input_data[numeric_cols].astype(float)

# DEBUG 
st.write("DEBUG INPUT:")
st.write(input_data)
st.write(input_data.dtypes)


# PREDIKSI
if st.button("Predict"):

    # Classification
    placement_pred = clf_model.predict(input_data)[0]

    # Regression
    salary_pred = reg_model.predict(input_data)[0]

    st.subheader("Hasil Prediksi")

    if placement_pred == 1:
        st.success("Mahasiswa kemungkinan DAPAT pekerjaan")
    else:
        st.error("Mahasiswa kemungkinan TIDAK mendapat pekerjaan")

    st.info(f"Estimasi Gaji: {salary_pred:.2f} LPA")
