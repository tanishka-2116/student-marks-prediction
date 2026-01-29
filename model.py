# Student Marks Prediction using Linear Regression
# This program predicts marks based on hours studied

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("data.csv")

# Separate input (Hours_Studied) and output (Marks)
X = data[["Hours_Studied"]]
y = data["Marks"]

# Create the Linear Regression model
model = LinearRegression()

# Train the model using the dataset
model.fit(X, y)

# Predict marks for a student who studied 6 hours
hours = [[6]]
predicted_marks = model.predict(hours)

print("Predicted marks for 6 hours of study:", predicted_marks[0])

