"""
This script includes all OS I/O methods and utilities
Most methods are wrappers for methods included in model.py and processing.py
"""

# OS I/O
import os
import pickle
import json

# Data handling
import pandas as pd

# Modeling (for type hinting only)
from sklearn.base import BaseEstimator

# Terminal output
from colorama import Fore, Style

# Internal modules
from toxikind.processing import fit_feature_scaler, transform_features
from toxikind.model import model_train, model_evaluate, model_predict

def fit_save_feature_scaler(path_X_train_raw: str,
                            path_feature_scaler: str
                           ) -> None:
    """
    This is a wrapper for "processing.fit_feature_scaler".
    It loads raw feature training data from given path,
    calls "processing.train_feature_scaler" and
    saves the trained scaler as .pkl.

    It assumes the raw data index column being unnamed.
    """
    # Laad raw feature training data from disk and set index
    X_train_raw = pd.read_csv(path_X_train_raw).set_index("Unnamed: 0")

    # Fit feature scaler
    feature_scaler = fit_feature_scaler(X_train_raw)

    # Save fitted scaler
    path_feature_scaler = f"{path_feature_scaler}/feature_scaler.pkl"
    with open(path_feature_scaler, "wb") as file:
        pickle.dump(feature_scaler, file)

    return None

def load_transform_save_features(path_feature_scaler: str,
                                 path_x_raw: str,
                                 path_x: str
                                ) -> None:
    """
    This is a wrapper for "processing.transform_features".
    It loads a fitted scaler from given path, raw feature data,
    calls "processing.transform_features" and
    saves the transformed data as .csv.

    It assumes the raw data index column being unnamed and
    renames it to "ID".
    """
    # Load feature_scaler
    path_feature_scaler = f"{path_feature_scaler}/feature_scaler.pkl"
    with open(path_feature_scaler, "rb") as file:
        feature_scaler = pickle.load(file)

    # Load data, rename index column, and set index
    X_raw = pd.read_csv(path_x_raw)
    X_raw = X_raw.rename(columns={"Unnamed: 0": "ID"}).set_index("ID")

    # Transform data
    X = transform_features(X_raw, feature_scaler)

    # Reset index and save data
    X = X.reset_index()
    X.to_csv(path_x, index=False)

def load_save_targets(path_y_raw: str,
                      path_y: str
                     ) -> None:
    """
    This function loads raw target data from given path, renames
    and saves it to given path.

    It assumes the raw data index column being unnamed and
    renames it to "ID".
    """
    y_train = pd.read_csv(path_y_raw).rename(columns={"Unnamed: 0": "ID"})
    y_train.to_csv(path_y, index=False)
    return None

def load_data(data_category: str) -> pd.DataFrame:
    """
    Purpose: load desired data
    - Check data category (X_train, y_train, X_test, y_test)
    - Select corresponding path, return error if not matched
    """
    pass

def check_valid_assay(assay: str) -> None:
    """
    Purpose: Check if assay is valid
    - Check if argument included in dict
    - Return error message and list of valid assays if not matched
    WARNING: many functions below depend on this one. Keep it above
    """

def train_save_model(assay: str) -> None:
    """
    Purpose: wrapper for model_train for one assay with saving
    - Load processed training data and call model_train
    - (NO train-val-split here)
    - Save new model as pickle
    """
    pass

def load_model(assay: str) -> BaseEstimator:
    """
    Purpose: Load the model for desired assay from hard drive

    WARNING: many functions below depend on this one. Keep it above
    """
    print(Fore.BLUE + f"\nLoading model from '{path_model}'..." + Style.RESET_ALL)

    with open(path_model, "rb") as f:
        model = pickle.load(f)

    print(Fore.GREEN + "✅ Model loaded successfully" + Style.RESET_ALL)
    return model

def save_model_metrics(assay: str)-> None:
    """
    Purpose: wrapper for model_evaluate with saving
    - Load model and processed testing data and call model_evaluate
    - Save metrics as JSON
    """
    pass

def save_model_prediction(assay: str)-> None:
    """
    Purpose: wrapper for model_predict with saving
    - Load model and call model_predict with new data to predict
    - Save preditctions as JSON
    """
    pass

def train_save_model_all(assays: dict) -> None:
    """
    Purpose: wrapper for retrain/refit all models
    - Access list of used models and loop above method through it.
    - Save all models
    """
    pass

def save_model_metrics_all(assays: dict)-> dict:
    """
    Purpose: wrapper for evaluate model and save metrics of all models
    - Access list of used models and loop above method through it.
    - Save all metrics
    """
    pass

def save_model_metrics_all(assays: dict)-> None:
    """
    Purpose: wrapper for evaluate model and save metrics of all models
    - Access list of used models and loop above method through it.
    - Save all metrics
    """
    pass

if __name__ == "__main__":
    # Fit/Save scaler
    path_X_train_raw = "raw_data/tox21_dense_train.csv.gz"
    path_feature_scaler = "production_model"
    fit_save_feature_scaler(path_X_train_raw, path_feature_scaler)

    # Load/Transform/Save training features
    path_feature_scaler = "production_model"
    path_x_raw = "raw_data/tox21_dense_train.csv.gz"
    path_x = "data/X_train.csv"
    load_transform_save_features(path_feature_scaler, path_x_raw, path_x)

    # Load/Save training targets
    path_y_raw = "raw_data/tox21_labels_train.csv.gz"
    path_y = "data/y_train.csv"
    load_save_targets(path_y_raw, path_y)

    # Load/Transform/Save testing features
    path_feature_scaler = "production_model"
    path_x_raw = "raw_data/tox21_dense_test.csv.gz"
    path_x = "data/X_test.csv"
    load_transform_save_features(path_feature_scaler, path_x_raw, path_x)

    # Load/Save testing targets
    path_y_raw = "raw_data/tox21_labels_test.csv.gz"
    path_y = "data/y_test.csv"
    load_save_targets(path_y_raw, path_y)

    # Train/save model
    path_model = "models"
    train_save_model()

    # Evaluate/save model
    path_model = "models"
    save_model_metrics()

    # Predict/save model
    path_model = "models"
    save_model_prediction()
