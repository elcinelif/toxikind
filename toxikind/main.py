"""
This script is to:
- (Re)Fit and save the feature scaler
- (Re)Create and save the preprocessed data
- Fast retrain of the production model for comparison
- Fast evaluation of the production model for comparison
"""

# OS I/O
import pickle

# Data handling
import pandas as pd

# Internal modules
from processing import fit_feature_scaler, transform_features

def fit_save_feature_scaler(path_X_train_raw: str,
                            path_feature_scaler: str
                           ) -> None:
    """
    This is a wrapper for "processing.fit_feature_scaler".
    It loads raw feature training data from given path,
    calls "processing.train_feature_scaler" and
    saves the trained scaler as .pickle.

    It assumes the raw data index column being unnamed.
    """
    # Laad raw feature training data from disk and set index
    X_train_raw = pd.read_csv(path_X_train_raw).set_index("Unnamed: 0")

    # Fit feature scaler
    feature_scaler = fit_feature_scaler(X_train_raw)

    # Save fitted scaler
    path_feature_scaler = f"{path_feature_scaler}/feature_scaler.pickle"
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
    path_feature_scaler = f"{path_feature_scaler}/feature_scaler.pickle"
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

def retrain_production_model() -> None:
    """
    Purpose: retrain/refit production model
    - Load processed training data
    - (NO train-val-split here)
    - Load production model
    - Retrain/refit production model
    - Store new model weights/params
    """
    pass

def evaluate_production_model()-> dict:
    """
    - Load retrained/refitted production model
    - Load processed testing data
    - Evaluate model and save metrics
    - Return metrics as dict
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

    # Retrain production model
    retrain_production_model()

    # Evaluate production model
    evaluate_production_model()
