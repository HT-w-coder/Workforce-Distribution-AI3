# update_employee_salary.py

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Employee.csv")

# Set base salaries depending on education
education_salary = {
    "Bachelors": 30000,
    "Masters": 45000,
    "PHD": 60000
}

# Salary modifiers
bench_penalty = 0.85  # if benched, reduce salary
experience_multiplier = 1.07  # 7% growth per year

# Generate Salary and Wage Data
base_salary = df["Education"].map(education_salary)
adjusted_salary = base_salary * (experience_multiplier ** df["ExperienceInCurrentDomain"])

# Apply bench penalty
adjusted_salary *= np.where(df["EverBenched"] == "Yes", bench_penalty, 1)

df["CurrentSalary"] = adjusted_salary.round(2)
df["ExpectedNextYearSalary"] = (df["CurrentSalary"] * experience_multiplier).round(2)
df["AnnualWageGrowth"] = (df["ExpectedNextYearSalary"] - df["CurrentSalary"]).round(2)

# Save to CSV
df.to_csv("Employee.csv", index=False)
print("✅ Salary fields added to Employee.csv")
