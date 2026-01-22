# Student Marks Prediction using Linear Regression

import pandas as pd
from sklearn.linear_model import LinearRegression

# load dataset
data = pd.read_csv("data.csv")

# input and output
X = data[["Hours_Studied"]]
y = data["Marks"]

# create and train model
model = LinearRegression()
model.fit(X, y)

# predict marks for a student who studied 6 hours
hours = [[6]]
predicted_marks = model.predict(hours)

print("Predicted marks for 6 hours of study:", predicted_marks[0])
