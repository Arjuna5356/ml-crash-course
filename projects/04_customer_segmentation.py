# Project 4: Customer Segmentation
# Goal: Group customers—like a mall cop with data powers.
# Dataset: Synthetic (6 samples, 2D points).
# Steps: Generate data, apply K-Means, analyze clusters.
# Real-World Use: Marketing, store layouts.
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [5, 6], [5, 8], [2, 1], [6, 7]])
model = KMeans(n_clusters=2, random_state=42)
model.fit(X)
print(f"Cluster labels: {model.labels_}")
print(f"Cluster centers: {model.cluster_centers_}")