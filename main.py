# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Load model, encoder, and accuracy
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")  # OneHotEncoder
accuracy = joblib.load("accuracy.pkl")

app = FastAPI()

# Allow frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class EmployeeInput(BaseModel):
    Education: str
    JoiningYear: int
    City: str
    PaymentTier: int
    Age: int
    Gender: str
    EverBenched: str
    ExperienceInCurrentDomain: int

@app.post("/predict")
def predict(data: EmployeeInput):
    try:
        # Convert input into a list of values
        categorical = [[
            data.Education,
            data.City,
            data.Gender,
            data.EverBenched
        ]]

        numerical = [
            data.JoiningYear,
            data.PaymentTier,
            data.Age,
            data.ExperienceInCurrentDomain
        ]

        # Transform categorical using OneHotEncoder
        encoded_categorical = encoder.transform(categorical)
        if hasattr(encoded_categorical, "toarray"):
            encoded_categorical = encoded_categorical.toarray()

        # Combine features
        final_input = np.concatenate([numerical, encoded_categorical[0]])

        # Prediction
        prediction = model.predict([final_input])[0]

        # Compute salary metrics
        base_salary = data.PaymentTier * 50000
        expected_growth = data.ExperienceInCurrentDomain * 0.05 * base_salary
        overall_wage = base_salary + expected_growth

        return {
            "leave_or_not": int(prediction),
            "expected_salary_growth": round(expected_growth, 2),
            "overall_wage": round(overall_wage, 2),
            "accuracy": round(accuracy * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}
