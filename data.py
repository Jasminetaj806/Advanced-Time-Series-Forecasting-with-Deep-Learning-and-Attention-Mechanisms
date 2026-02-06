import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

n_steps = 1200
time = np.arange(n_steps)

data = pd.DataFrame({
    "feature_1": np.sin(0.02 * time) + np.random.normal(0, 0.1, n_steps),
    "feature_2": np.cos(0.015 * time) + np.random.normal(0, 0.1, n_steps),
    "feature_3": np.sin(0.05 * time) + np.random.normal(0, 0.1, n_steps),
    "feature_4": np.random.normal(0, 0.3, n_steps),
    "target": np.sin(0.02 * time) + 0.5*np.cos(0.05 * time) + np.random.normal(0, 0.1, n_steps)
})

print(data.head())
