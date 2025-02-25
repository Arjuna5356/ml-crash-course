# Project 5: Spam Email Detector
# Goal: Zap spam emails—like a digital exterminator.
# Dataset: Synthetic (4 text samples, spam or not).
# Steps: Vectorize text, train Naive Bayes, predict.
# Real-World Use: Email filters, chat moderation.
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
texts = ["win cash now", "meeting tomorrow", "free prize", "hi friend"]
labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)
test = vectorizer.transform(["win free stuff"])
print(f"Spam? {model.predict(test)[0]}")  # 1 = Yup, spam!