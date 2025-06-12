# Data handling
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

def fit_feature_scaler(X_train_raw: pd.DataFrame) -> Pipeline:
    """
    This function fits a MinMaxScaler with raw feature training data
    """
    # ColumnTransformer with MinMaxScaler
    column_transformer = ColumnTransformer([
        ("minmax_scaler", MinMaxScaler(), X_train_raw.columns)
        ])

    # Pipeline
    pipeline = Pipeline([
        ("column_transformer", column_transformer)
        ])

    # Fit feature_scaler with raw feature training data
    print("Fitting scaler...")
    feature_scaler = pipeline.fit(X_train_raw)

    # Return fitted feature_scaler
    print("✅ Scaler fitted")
    return feature_scaler

def transform_features(X_raw: pd.DataFrame, feature_scaler: Pipeline) -> pd.DataFrame:
    """
    This function transforms features using a scaler fitted on raw feature training data
    """
    # Transform features
    print("Transforming features...")
    X = pd.DataFrame(feature_scaler.transform(X_raw), columns=X_raw.columns, index=X_raw.index)

    # Return transformed data
    print("✅ Features transformed")
    return X
