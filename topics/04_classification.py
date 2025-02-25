# Topic 4: Classification
# Objective: Sort things into categories—like a librarian on steroids.
# What You’ll Learn: Logistic regression for binary decisions.
# Real-World Use: Spam filters, disease diagnosis, or cat vs. dog pics.
# Tip: Balance your classes—too many “yes”es skews it!
from sklearn.linear_model import LogisticRegression
X = [[1], [2], [3], [4]]  # Hours studied
y = [0, 0, 1, 1]          # 0 = Fail, 1 = Pass
model = LogisticRegression()
model.fit(X, y)
print(f"Pass with 2.5 hrs? {model.predict([[2.5]])[0]}")