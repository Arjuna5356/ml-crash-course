# Project 3: Handwritten Digit Recognizer
# Goal: Decode scribbles—like a teacher grading doodles.
# Dataset: Digits (sklearn, 1797 samples, 8x8 images).
# Steps: Load digits, split, train a neural net, test it.
# Real-World Use: OCR, postal services.
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test)}")  # ~0.95+