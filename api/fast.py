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
def predict(target='Please choose test: SR.MMP or SR.ARE', compound='Please type in a compound!'):
    # Check if compound in test data
    if compound == 'Please type in a compound!':
        return compound

    X_test = pd.read_csv('data/X_test.csv',index_col='ID')
    X_predict = X_test[X_test.index==compound]

    if X_predict.shape[0] == 0:
        return 'Compound not found in Dataset'

    # Check selected target and load model
    if target == 'SR.MMP':
        with open('production_model/fitted_xgb_mmp.pkl','rb') as f:
            trained_model = pickle.load(f)
    elif target == 'SR.ARE':
        with open('production_model/fitted_xgb_are.pkl','rb') as f:
            trained_model = pickle.load(f)
    else:
        return 'Please choose test: SR.MMP or SR.ARE'

    # Predict and return
    prediction = int(trained_model.predict(X_predict)[0])

    if prediction == 1:
        return {'Test type': target,
                 compound: 'This compound is toxic'}

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
