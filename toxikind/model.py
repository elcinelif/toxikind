# Data handling
import numpy as np
import pandas as pd

# Modeling
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate

# Terminal output
from colorama import Fore, Style

def prepare_data(assay: str,
                 X: pd.DataFrame,
                 y: pd.DataFrame) -> pd.DataFrame:
    """
    Selects desired assay from target data, removes NaNs
    and corresponding feature data rows.

    Parameters:
    - assay: desired assay
    - X: feature data
    - y: target data

    Returns:
    - feature and target data as pd.DataFrames
    """
    print(Fore.BLUE + "\nPreparing data..." + Style.RESET_ALL)
    # Create single target for desired assay without NAs
    y_assay = y[[assay]].dropna()

    # Inner merge with features
    data_assay = y_assay.merge(X, how="inner", on="ID")

    # Split again
    X_assay = data_assay.drop(columns=[assay])
    y_assay = data_assay[assay]
    print(Fore.GREEN + "✅ Data prepared!" + Style.RESET_ALL)

    # Return both DataFrames
    return X_assay, y_assay

def model_train(assay: str,
                X_train: pd.DataFrame,
                y_train: pd.DataFrame) -> BaseEstimator:
    """
    Trains model for desired assay.

    Parameters:
    - assay: desired assay
    - X_train: training feature data
    - y_train: training target data

    Returns:
    - Model

    Note:
    - Assay argument check called from wrapper function.
    - model hyperparameters are hardcoded.

    """
    # Preparation
    X_train_assay, y_train_assay = prepare_data(assay, X_train, y_train)

    # Hyperparameters
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
    n_estimators = 300
    learning_rate = 0.1
    max_depth = 4

    # Definition and Instantiation
    model_GBC = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
        )

    # Fit
    model_GBC.fit(X_train_assay, y_train_assay)

    # Return model
    print(Fore.GREEN + "✅ Model trained!" + Style.RESET_ALL)
    return model_GBC

def model_evaluate(model: BaseEstimator,
                   assay: str,
                   X: pd.DataFrame,
                   y: pd.DataFrame,
                   cv: int=5) -> dict:
    """
    Cross-evaluates the most recent model for desired assay.

    Parameters:
    - model: a model
    - X: feature data
    - y: target data

    Returns:
    - Model metrics as a dictionary
    """
    # Preparation
    X_assay, y_assay = prepare_data(assay, X, y)

    # Cross-validate
    print(Fore.BLUE + "\nEvaluating model..." + Style.RESET_ALL)
    model_cv = cross_validate(model, X=X_assay, y=y_assay, cv=cv,
                              scoring=["accuracy", "recall", "precision", "f1"]
                             )

    # Cast validation metrics into dictionary
    model_metrics = pd.DataFrame((list(model_cv.values()))[2:]).mean(axis=1)
    keys = ["accuracy", "recall", "precision", "f1"]
    model_metrics = dict(zip(keys, list(model_metrics)))
    print(Fore.GREEN + "✅ Model evaluated!" + Style.RESET_ALL)
    return model_metrics

def model_predict(model: BaseEstimator,
                  X: np.ndarray,
                  threshold: float=0.2627) -> np.ndarray:
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
    # Compute probabilities
    probs = model.predict_proba(X)
    toxic_probs = probs[:, 1]

    # Filter using trheshold and return
    predictions = (toxic_probs >= threshold).astype(int)
    print(Fore.GREEN + "✅ Predictions completed" + Style.RESET_ALL)
    return predictions
