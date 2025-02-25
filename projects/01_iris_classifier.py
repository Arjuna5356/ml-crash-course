# Project 1: Iris Flower Classifier
# Goal: Predict flower species—like a botanist with a PhD in cool.
# Dataset: Iris (built-in sklearn, 150 samples, 4 features, 3 classes).
# Steps: Load data, split it, train a Random Forest, check accuracy.
# Real-World Use: Species ID, quality control.
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test)}")  # ~0.9+ usually