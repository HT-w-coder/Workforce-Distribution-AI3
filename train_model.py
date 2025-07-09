# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the updated dataset
df = pd.read_csv("Employee.csv")

# Define target and use only numerical features
X = df.select_dtypes(include=['int64', 'float64']).drop(columns=["LeaveOrNot"])
y = df["LeaveOrNot"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved.")
