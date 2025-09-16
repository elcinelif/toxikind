import os

##################  ENVIRONMENT VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")

# Docker local
DOCKER_IMAGE_NAME = os.environ.get("DOCKER_IMAGE_NAME")
DOCKER_LOCAL_PORT = os.environ.get("DOCKER_LOCAL_PORT")

# Google Cloud
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")

# Docker Artifact Registry & Google Cloud Run
DOCKER_REPO_NAME = os.environ.get("DOCKER_REPO_NAME")
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")

##################  CONSTANTS  ##################
PATH_X_TRAIN_RAW = "raw_data/tox21_dense_train.csv.gz"
PATH_FEATURE_SCALER = "production_model"
PATH_X_TRAIN = "data/X_train.csv"
PATH_X_TEST_RAW = "raw_data/tox21_dense_test.csv.gz"
PATH_X_TEST = "data/X_test.csv"
PATH_Y_TRAIN_RAW = "raw_data/tox21_labels_train.csv.gz"
PATH_Y_TRAIN = "data/y_train.csv"
PATH_Y_TEST_RAW = "raw_data/tox21_labels_test.csv.gz"
PATH_Y_TEST = "data/y_test.csv"
BASE_PATH_DATA = "data"
BASE_PATH_MODEL = "models"

################## MODEL HYPERPARAMETERS ##################

################## VALIDATIONS ##################
VALID_ASSAYS = {
                "NR.AhR": "ahr",
                "NR.AR": "ar",
                "SR.ARE": "are",
                "NR.ER.LBD": "er_lbd",
                "NR.AR.LBD": "ar_lbd", #Inaccurate in the notebook ("lbd")!
                # This WILL cause incomp with fast.py to be fixed!
                "SR.MMP": "mmp"
               }
