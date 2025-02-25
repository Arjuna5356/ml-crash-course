# Topic 5: Decision Trees
# Objective: Make decisions with a flowchart-like model.
# What You’ll Learn: Splitting data based on rules.
# Real-World Use: Loan approvals, game AI, or fruit sorting.
# Tip: Don’t let it grow too wild—prune it!
from sklearn.tree import DecisionTreeClassifier
X = [[1, 2], [2, 3], [3, 1]]  # [weight, sweetness]
y = [0, 1, 0]                  # 0 = Apple, 1 = Orange
model = DecisionTreeClassifier()
model.fit(X, y)
print(f"Fruit at [2, 2]: {model.predict([[2, 2]])[0]}")