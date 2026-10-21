import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


# Example dataset (replace later with real collected logs)
X = np.array([
    [0.85, 0.80, 0.9, 0.8],
    [0.40, 0.35, 0.6, 0.5],
    [0.90, 0.70, 0.95, 0.8],
    [0.20, 0.30, 0.4, 0.5],
])

y = np.array([1, 0, 1, 0])  # 1 = GRANTED, 0 = DENIED


model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "data/embeddings/fusion_model.pkl")

print("[INFO] Fusion model trained and saved.")