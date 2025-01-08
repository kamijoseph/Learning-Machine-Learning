
# Simple Logistic regresion algorithm using synthetic data.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generating the synthetic data and 500 samples with 2 features
np.random.seed(42)
n_samples = 500 

# Feature
X1 = np.random.normal(loc=2.5, scale=1.0, size=n_samples)
X2 = np.random.normal(loc=1.5, scale=1.5, size=n_samples)

# Decision boundary
y = (2.5 * X1 - 1.5 * X2 > 2.5).astype(int)

# Combining the features
X = np.column_stack((X1, X2))

# Upload and Save te data into a csv file
data = pd.DataFrame(data={"Feature1": X1, "Feature2": X2, "Target": y})
data.to_csv("synthetic_data1.csv", index=False)
print("Data saved to synthetic_data1.csv")