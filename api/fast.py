from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle

app = FastAPI()
#app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(target='SR.MMP', compound='Please Type in a compound!'):

    if compound == 'Please Type in a compound!':
        return compound

    X_test = pd.read_csv('data/X_test.csv',index_col='ID')
    X_predict = X_test[X_test.index==compound]

    if X_predict.shape[0] == 0:
        return 'Compound not found in Dataset'

    with open('production_model/gb_model.pkl','rb') as f:
        trained_model = pickle.load(f)

    with open('production_model/feature_scaler.pkl','rb') as f:
        scaler_fitted = pickle.load(f)

    X_predict_transformed = pd.DataFrame(scaler_fitted.transform(X_predict),columns=X_predict.columns)

    prediction = int(trained_model.predict(X_predict_transformed)[0])

    if prediction == 1:
        return 'The compound is toxic'

    return {'Test type': target,
            compound: 'This compound is not toxic'}

@app.get("/")
def root():
    return {"Purpose": "This is a toxicity predictor"}


if __name__ == '__main__':
    try:
        predict("NCGC00261900-01")
    except:
        import ipdb, traceback, sys

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
