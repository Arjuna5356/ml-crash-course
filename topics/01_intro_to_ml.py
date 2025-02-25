# Topic 1: Introduction to Machine Learning
# Objective: Understand what ML is and how it mimics human learning.
# What You’ll Learn: Basics of supervised learning with a simple linear model.
# Real-World Use: Predicting sales, stock prices, or even your cat’s mood.
# Tip: Start small—ML is just fancy guesswork with data!
from sklearn.linear_model import LinearRegression
import numpy as np
X = np.array([[1], [2], [3], [4]])  # Input: x values
y = np.array([2, 4, 6, 8])          # Output: y = 2x
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[5]])
print(f"Predicting for x=5: {prediction[0]}")  # Should be ~10