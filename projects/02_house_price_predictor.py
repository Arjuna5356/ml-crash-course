# Project 2: House Price Predictor
# Goal: Guess house prices—because who doesn’t want to be rich?
# Dataset: Synthetic (5 samples, size vs. price).
# Steps: Create data, split, train Linear Regression, predict.
# Real-World Use: Real estate, budgeting.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
X = np.array([[1000], [1500], [2000], [2500], [3000]])
y = np.array([200000, 300000, 400000, 500000, 600000])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print(f"Price for 1750 sq ft: {model.predict([[1750]])[0]}")