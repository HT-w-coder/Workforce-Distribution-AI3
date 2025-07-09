import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and encoder
model = joblib.load("model.pkl")

# Load dataset with salary info
df = pd.read_csv("Employee.csv")

st.title("🌟 Workforce Distribution AI")
st.write("Predict employee retention and analyze salary growth")

# Sidebar Inputs
st.sidebar.header("Employee Information")
gender = st.sidebar.selectbox("Gender", df["Gender"].unique())
ever_benched = st.sidebar.selectbox("Ever Benched", df["EverBenched"].unique())
city = st.sidebar.selectbox("City", df["City"].unique())
education = st.sidebar.selectbox("Education", df["Education"].unique())
joining_year = st.sidebar.slider("Joining Year", int(df["JoiningYear"].min()), int(df["JoiningYear"].max()))
payment_tier = st.sidebar.selectbox("Payment Tier", sorted(df["PaymentTier"].unique()))
age = st.sidebar.slider("Age", int(df["Age"].min()), int(df["Age"].max()))
experience = st.sidebar.slider("Experience in Current Domain", int(df["ExperienceInCurrentDomain"].min()), int(df["ExperienceInCurrentDomain"].max()))

# Construct input DataFrame
input_data = pd.DataFrame({
    "Gender": [gender],
    "EverBenched": [ever_benched],
    "City": [city],
    "Education": [education],
    "JoiningYear": [joining_year],
    "PaymentTier": [payment_tier],
    "Age": [age],
    "ExperienceInCurrentDomain": [experience]
})

# Prediction
prediction = model.predict(input_data)[0]
st.subheader("Prediction")
st.success("✅ Will Stay" if prediction == 1 else "❌ Will Leave")

# Salary Info Display
if "Salary" in df.columns and "OverallWage" in df.columns:
    st.subheader("💰 Salary Analysis")

    # Find employees with same profile
    matched = df[
        (df["Gender"] == gender) &
        (df["Education"] == education) &
        (df["EverBenched"] == ever_benched)
    ]

    # Plot average salary vs experience
    growth_data = matched.groupby("ExperienceInCurrentDomain")["Salary"].mean().reset_index()
    plt.figure(figsize=(8, 4))
    plt.plot(growth_data["ExperienceInCurrentDomain"], growth_data["Salary"], marker='o')
    plt.title("Salary Growth with Experience")
    plt.xlabel("Experience (Years)")
    plt.ylabel("Average Salary")
    st.pyplot(plt)

    # Show overall wage info
    st.write("### Overall Wage Distribution")
    st.bar_chart(df.groupby("Education")["OverallWage"].mean())
else:
    st.warning("Salary or OverallWage data not found in the dataset. Please update Employee.csv.")
