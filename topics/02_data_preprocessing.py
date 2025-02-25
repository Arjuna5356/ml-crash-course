# Topic 2: Data Preprocessing
# Objective: Clean and prep data for ML—like grooming a scruffy dog.
# What You’ll Learn: Scaling features to make models happy.
# Real-World Use: Fixing messy customer data for better predictions.
# Tip: Garbage in, garbage out—clean data is king!
from sklearn.preprocessing import StandardScaler
import numpy as np
data = np.array([[1], [2], [10], [100]])  # Raw, wild data
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
print(f"Scaled data (mean=0, std=1): {scaled}")