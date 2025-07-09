# main2.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and dataset
model = joblib.load("model.pkl")
df = pd.read_csv("Employee.csv")

st.title("📊 Workforce Distribution AI")
st.subheader("🔍 Predict Employee Retention and Visualize Salary Growth")

# Input fields
joining_year = st.number_input("Joining Year", min_value=2000, max_value=2025, value=2015)
payment_tier = st.number_input("Payment Tier", min_value=1, max_value=3, value=1)
age = st.slider("Age", min_value=18, max_value=60, value=30)
experience = st.slider("Experience in Current Domain", 0, 20, 3)
current_salary = st.number_input("Current Salary", value=40000)
expected_next_year = st.number_input("Expected Salary Next Year", value=42800)

# Feature vector
input_data = pd.DataFrame([{
    "JoiningYear": joining_year,
    "PaymentTier": payment_tier,
    "Age": age,
    "ExperienceInCurrentDomain": experience,
    "CurrentSalary": current_salary,
    "ExpectedNextYearSalary": expected_next_year,
    "AnnualWageGrowth": expected_next_year - current_salary
}])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Will Leave ❌" if prediction == 1 else "Will Stay ✅"
    st.success(f"Prediction: **{result}**")

# Visualization
st.subheader("📈 Salary Growth Trend")

# Average salary growth by experience
avg_growth = df.groupby("ExperienceInCurrentDomain")["AnnualWageGrowth"].mean()

fig, ax = plt.subplots()
avg_growth.plot(kind='line', marker='o', ax=ax)
ax.set_title("Average Annual Wage Growth by Experience")
ax.set_xlabel("Experience (Years)")
ax.set_ylabel("Annual Wage Growth")
st.pyplot(fig)
