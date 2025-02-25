# Topic 6: Random Forests
# Objective: Boost accuracy with a team of trees.
# What You’ll Learn: Ensemble learning power.
# Real-World Use: Fraud detection, medical diagnosis, or movie ratings.
# Tip: More trees = better, but slower—find the sweet spot!
from sklearn.ensemble import RandomForestClassifier
X = [[1, 0], [2, 1], [3, 0]]  # [age, coffee]
y = [0, 1, 0]                  # 0 = Sleepy, 1 = Awake
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)
print(f"Awake at [2, 0]? {model.predict([[2, 0]])[0]}")