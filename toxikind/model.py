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
    Predict using a scikit-learn model.
    """
    print(Fore.BLUE + "\nMaking predictions..." + Style.RESET_ALL)

    predictions = model.predict(X)

    print(Fore.GREEN + "✅ Predictions completed" + Style.RESET_ALL)
    return predictions
