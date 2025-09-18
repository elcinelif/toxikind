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

# Modeling (in this file for type hinting only)
from sklearn.base import BaseEstimator

# Terminal output
from colorama import Fore, Style

# Internal modules
import toxikind.params as params
from toxikind.processing import fit_feature_scaler, transform_features
from toxikind.model import model_train, model_evaluate, model_predict

def fit_save_feature_scaler(path_X_train_raw: str=params.PATH_X_TRAIN_RAW,
                            path_feature_scaler: str=params.PATH_FEATURE_SCALER
                           ) -> None:
    """
    This is a wrapper for "processing.fit_feature_scaler".

    Parameters:
    - path_X_train_raw: path to raw training feature data
    - path_feature_scaler: path to store trained feature scaler

    Returns:
    - A feature scaler in .pkl format to hard drive
    - None to namespace

    Note: assumes the raw data index column being unnamed.
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

def load_transform_save_features(data: str,
                                 path_x_raw: str=None,
                                 path_x: str=None,
                                 path_feature_scaler: str=params.PATH_FEATURE_SCALER
                                ) -> None:
    """
    Wrapper for "processing.transform_features" with hard drive operations
    and feature data type selection.

    Parameters:
    - path_feature_scaler: path to store trained feature scaler
    - path_X_raw: path to raw feature data
    - path_x: path to save transformed feature data

    Returns:
    - Transformed feature data in .csv format to hard drive
    - None to namespace

    Note: assumes the raw data index column being unnamed and
    renames it to "ID".
    """
    # Match desired feature data type with corresponding path
    if path_x_raw is None and path_x is None: #Causes error if only one is set
        match data:
            case "train":
                path_x_raw = params.PATH_X_TRAIN_RAW
                path_x = params.PATH_X_TRAIN
            case "test":
                path_x_raw = params.PATH_X_TEST_RAW
                path_x = params.PATH_X_TEST
            case _:
                error_msg = f"❌ '{data}' is not a valid option. Choose either 'train' or 'test'"
                print(Fore.RED + error_msg + Style.RESET_ALL)
                raise ValueError(error_msg)
    else:
        print(Fore.YELLOW + "⚠️ Paths already set. Skipping reassignment." + Style.RESET_ALL)

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

def load_transform_save_targets(data: str,
                                path_y_raw: str=None,
                                path_y: str=None
                               ) -> None:
    """
    Loads raw target data for selected data type. Due to the simplicity of operations
    this is the transformer und wrapper for hard drive operations in one function.

    Arguments:
    - path_y_raw: path to raw target data
    - path_y: path to store renamed target data

    Returns:
    - Renamed target data in .csv format to hard drive
    - None to namespace

    Note: assumes the raw data index column being unnamed and
    renames it to "ID".
    """
    # Match desired feature data type with corresponding path
    if path_y_raw is None and path_y is None: #Causes error if only one is set
        match data:
            case "train":
                path_y_raw = params.PATH_Y_TRAIN_RAW
                path_y = params.PATH_Y_TRAIN
            case "test":
                path_y_raw = params.PATH_Y_TEST_RAW
                path_y = params.PATH_Y_TEST
            case _:
                error_msg = f"❌ '{data}' is not a valid option. Choose either 'train' or 'test'"
                print(Fore.RED + error_msg + Style.RESET_ALL)
                raise ValueError(error_msg)
    else:
        print(Fore.YELLOW + "⚠️ Paths already set. Skipping reassignment." + Style.RESET_ALL)

    # Transform targets
    print("Transforming targets...")
    y_train = pd.read_csv(path_y_raw).rename(columns={"Unnamed: 0": "ID"})
    y_train.to_csv(path_y, index=False)
    print("✅ Targets transformed")
    return None

def load_data(data: str,
              base_path_data: str=params.BASE_PATH_DATA,
             ) -> pd.DataFrame:
    """
    Loads preprocessed data.

    Parameters:
    - base_path_data: path to data folder
    - data: type of data to be loaded

    Returns:
    - Data

    Raises:
    - ValueError if type of data is not valid

    Note: has several dependencies
    """
    # Dict of valid arguments for "data" parameter.
    valid_arguments = { #This stays here because it is short and static
        "X_train": "X_train.csv",
        "y_train": "y_train.csv",
        "X_test": "X_test.csv",
        "y_test": "y_test.csv"
        }

    # Check if the key is valid
    if data not in valid_arguments:
        error_msg = f"❌ '{data}' is not a valid option. Choose from: {', '.join(valid_arguments.keys())}"
        print(Fore.RED + error_msg + Style.RESET_ALL)
        raise ValueError(error_msg)

    # Load data
    full_path_data = os.path.normpath(os.path.join(base_path_data, valid_arguments.get(data)))
    df = pd.read_csv(full_path_data).set_index("ID")
    return df

def check_valid_assay(assay: str,
                      valid_assays: dict = params.VALID_ASSAYS) -> None:
    """
    Checks if assay is valid.

    Parameters:
    - assay: desired assay
    - valid_assays: dict of valid assays

    Returns:
    - None

    Raises:
    - ValueError if desired assay is not valid

    Note: has several dependencies
    """
    if assay not in valid_assays:
        error_msg = f"❌ '{assay}' is not valid. Choose from: {', '.join(valid_assays.keys())}"
        print(Fore.RED + error_msg + Style.RESET_ALL)
        raise ValueError(error_msg)
    return None

def train_save_model(assay: str,
                     base_path_model: str = params.BASE_PATH_MODEL) -> None:
    """
    This is a wrapper for "model.model_train" with saving.
    Includes assay validity check.

    Parameters:
    - assay: desired assay
    - base_path_model: path to save model to

    Returns:
    - A model in .pkl format to hard drive
    - None to namespace

    Note: model path is different from production model path, unlike the path
    of the feature scaler. Moving models to production happens manually with OS.
    """
    # Check assay argument
    # Load data from hard drive
    # Call model.model_train
    # Save model to hard drive
    pass

def load_model(assay: str) -> BaseEstimator:
    """
    Purpose: Load the model for desired assay from hard drive

    WARNING: many functions below depend on this one. Keep it above
    """
    #print(Fore.BLUE + f"\nLoading model from '{path_model}'..." + Style.RESET_ALL)

    #with open(path_model, "rb") as f:
    #    model = pickle.load(f)

    #print(Fore.GREEN + "✅ Model loaded successfully" + Style.RESET_ALL)
    #return model
    pass

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

if __name__ == "__main__":
    import sys
    match sys.argv[1]:
        case "run_fit_save_feature_scaler":
            fit_save_feature_scaler()
        case "run_load_transform_save_train_features":
            load_transform_save_features("train")
        case "run_load_transform_save_test_features":
            load_transform_save_features("test")
        case "run_load_transform_save_valid_features":
            load_transform_save_features("valid")
        case "run_load_transform_save_train_targets":
            load_transform_save_targets("train")
        case "run_load_transform_save_test_targets":
            load_transform_save_targets("test")
        case "run_load_transform_save_valid_targets":
            load_transform_save_features("valid")
        case "run_show_train_features":
            print(load_data("X_train").head())
        case "run_show_test_features":
            print(load_data("X_test").head())
        case "run_show_train_targets":
            print(load_data("y_train").head())
        case "run_show_test_targets":
            print(load_data("y_test").head())
        case "run_show_invalid_data_input":
            print(load_data("Z-vor!").head())
        case _:
            print("Unknown command.")
