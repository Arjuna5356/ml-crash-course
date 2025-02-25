# Topic 10: Model Evaluation
# Objective: Judge your model’s skills—like a talent show.
# What You’ll Learn: Accuracy, precision, and more.
# Real-World Use: Comparing models, avoiding overhype.
# Tip: Test on unseen data—don’t cheat!
from sklearn.metrics import accuracy_score
y_true = [0, 1, 0, 1]
y_pred = [0, 1, 1, 1]
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")  # 0.75—decent!