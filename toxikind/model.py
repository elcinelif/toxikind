import pickle
import numpy as np
from colorama import Fore, Style

def load_model(path: str):
    """
    Load a scikit-learn model from a pickle (.pkl) file.
    """
    print(Fore.BLUE + f"\nLoading model from '{path}'..." + Style.RESET_ALL)

    with open(path, "rb") as f:
        model = pickle.load(f)

    print(Fore.GREEN + "✅ Model loaded successfully" + Style.RESET_ALL)
    return model


def predict_model(model, X: np.ndarray):
    """
    Predict using a scikit-learn model with a custom probability threshold.

    Parameters:
        model: A fitted scikit-learn classifier with predict_proba()
        X: Feature matrix (2D numpy array)
        threshold: Classification threshold for positive class (default 0.2627)

    Returns:
        A numpy array of binary predictions (0 or 1)
    """
    print(Fore.BLUE + "\nMaking predictions..." + Style.RESET_ALL)

    probs = model.predict_proba(X)
    toxic_probs = probs[:, 1]

    predictions = (toxic_probs >= 0.2627).astype(int)

    print(Fore.GREEN + "✅ Predictions completed" + Style.RESET_ALL)

    return predictions
