import numpy as np
import pandas as pd

def generate_data(n_steps=1200):
    t = np.arange(n_steps)
    trend = t * 0.005
    season1 = np.sin(2 * np.pi * t / 50)
    season2 = np.sin(2 * np.pi * t / 100)
    noise = np.random.normal(0, 0.2, n_steps)
    series = trend + season1 + season2 + noise
    return pd.DataFrame({"value": series})

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("synthetic_series.csv", index=False)
    print("synthetic_series.csv created")