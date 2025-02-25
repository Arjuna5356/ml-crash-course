# Topic 3: Regression
# Objective: Predict continuous numbers—like a crystal ball with math.
# What You’ll Learn: Linear regression for trends.
# Real-World Use: House prices, temperature forecasts, or your grocery bill.
# Tip: Look for linear patterns in your data first!
from sklearn.linear_model import LinearRegression
X = [[1000], [1500], [2000]]  # House sizes (sq ft)
y = [200000, 300000, 400000]  # Prices
model = LinearRegression()
model.fit(X, y)
print(f"Price for 1750 sq ft: {model.predict([[1750]])[0]}")