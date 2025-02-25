# Topic 7: Clustering
# Objective: Group data without hints—like a blind taste test.
# What You’ll Learn: K-Means for unsupervised learning.
# Real-World Use: Customer segments, image compression, or party cliques.
# Tip: Guess the number of clusters—trial and error FTW!
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [5, 6], [5, 8]])
model = KMeans(n_clusters=2)
model.fit(X)
print(f"Clusters: {model.labels_}")