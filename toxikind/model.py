# Data handling
import numpy as np
import pandas as pd

# Modeling
from sklearn.base import BaseEstimator

# Terminal output
from colorama import Fore, Style

def model_train(assay: str,
                X_train: pd.DataFrame,
                y_train: pd.DataFrame) -> BaseEstimator:
    """
    Purpose: select assay and train corresponding model
    - (Valid assay check is done in main.py)
    - Use assay argument to train on desired target
    - Model and its parameters are hard-coded as of now
    """
    pass

def model_evaluate(model: BaseEstimator,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> dict:
    """
    Purpose: evaluate given model and return metrics
    - (model selection is in main.py)
    - Print metrics and return as dict
    - (Save as JSON in main.py)
    """
    pass

def model_predict(model: BaseEstimator,
                  X: np.ndarray) -> np.ndarray:
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
