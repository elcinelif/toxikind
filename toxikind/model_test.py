from model import load_model, predict_model
import numpy as np
import os

# Construct the path relative to main.py
model_path = os.path.join(os.path.dirname(__file__), "GradientBoosting.pkl")

# Load the model
model = load_model(model_path)
X_sample = np.random.rand(1, 801)


predictions = predict_model(model, X_sample)
print(predictions)
