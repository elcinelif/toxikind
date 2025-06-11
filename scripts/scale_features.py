# OS I/O
import os

# Data handling
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    This function scales feature tables with a MinMaxScaler trained on the input data
    """
    # ColumnTransformer with MinMaxScaler
    column_transformer = ColumnTransformer([
        ("minmax_scaler", MinMaxScaler(), X.columns)
        ])

    # Pipeline
    pipeline = Pipeline([
        ("column_transformer", column_transformer)
        ])

    # Transform data while preserving column names and index
    X = pd.DataFrame(pipeline.fit_transform(X), columns=X.columns, index=X.index)

    print("✅ features processed.")

    return X

if __name__ == '__main__':
    # Load/Scale/Save training features
    path_features_train = "../raw_data/tox21_dense_train.csv.gz"
    X_train_raw = pd.read_csv(path_features_train).set_index("Unnamed: 0")
    X_train = scale_features(X_train_raw)
    X_train.to_csv("../data/X_train.csv")

    # Load/Scale/Save testing features
    path_features_test = "../raw_data/tox21_dense_test.csv.gz"
    X_test_raw = pd.read_csv(path_features_test).set_index("Unnamed: 0")
    X_test = scale_features(X_test_raw)
    X_test.to_csv("../data/X_test.csv")
