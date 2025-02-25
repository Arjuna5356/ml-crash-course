# Topic 8: PCA (Dimensionality Reduction)
# Objective: Simplify data without losing the plot.
# What You’ll Learn: Reducing features with PCA.
# Real-World Use: Speeding up models, visualizing high-D data.
# Tip: Keep enough variance—80% is a good start!
from sklearn.decomposition import PCA
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6]])
pca = PCA(n_components=1)
reduced = pca.fit_transform(X)
print(f"Reduced to 1D: {reduced}")