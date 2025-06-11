"""
This script is to:
- fast recreate scaled data if needed
- fast retrain and reevaluate the current production model for comparison
  with models in development
"""

# Data handling
import pandas as pd

# Internal modules
from scale_features import scale_features

def preprocess_train_data() -> None:
    """
    Purpose: preprocess raw training data (if needed)
    Steps:
    - Load raw training data
    - Scale (minmax) features using "scale_features.py"
    - Save processed training data in different folder
    """
    # Features
    path_features_train = "../raw_data/tox21_dense_train.csv.gz"
    X_train_raw = pd.read_csv(path_features_train).set_index("Unnamed: 0")
    X_train = scale_features(X_train_raw)
    X_train.to_csv("../data/X_train.csv")

    # Targets
    path_targets_train = "../raw_data/tox21_labels_train.csv.gz"
    y_train = pd.read_csv(path_targets_train).set_index("Unnamed: 0")
    y_train.to_csv("../data/y_train.csv")

def preprocess_test_data() -> None:
    """
    Purpose: preprocess raw testing data (if needed)
    Steps:
    - Load raw testing data
    - Scale (minmax) features
    - Save processed testing data in different folder
    """
    # Features
    path_features_test = "../raw_data/tox21_dense_test.csv.gz"
    X_test_raw = pd.read_csv(path_features_test).set_index("Unnamed: 0")
    X_test = scale_features(X_test_raw)
    X_test.to_csv("../data/X_test.csv")

    # Targets
    path_targets_test = "../raw_data/tox21_labels_test.csv.gz"
    y_test = pd.read_csv(path_targets_test).set_index("Unnamed: 0")
    y_test.to_csv("../data/y_test.csv")

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


def reevaluate_production_model()-> dict:
    """
    - Load retrained/refitted production model
    - Load processed testing data
    - Evaluate model and save metrics
    - Return metrics as dict
    """
    pass

if __name__ == '__main__':
    preprocess_train_data()
    preprocess_test_data()
    retrain_production_model()
    reevaluate_production_model()
