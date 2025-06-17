from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
from pydantic import BaseModel

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
def predict(compound='Please type in a compound!'):
    # Check if compound in test data
    if compound == 'Please type in a compound!':
        return compound
    #Featurization of compound
    X_test = pd.read_csv('data/X_test.csv',index_col='ID')
    X_predict = X_test[X_test.index==compound]

    if X_predict.shape[0] == 0:
        return 'Compound not found in Dataset'

    model_list = ['fitted_xgb_are','fitted_xgb_mmp']
    target_list = ['ARE', 'MMP']
    predictions = []
    predictions_probabilities = []

    for i in model_list:

        with open(f'production_model/{i}.pkl','rb') as f:
            trained_model= pickle.load(f)

        prediction = trained_model.predict(X_predict)[0]
        prediction_probability = f'{round(trained_model.predict_proba(X_predict)[:,-1][0]*100,2)}%'

        predictions.append(prediction)
        predictions_probabilities.append(prediction_probability)

    results_df = pd.DataFrame({
    "target": target_list,
    "prediction": predictions,
    "probability": predictions_probabilities})

    results_df['result'] = results_df['prediction'].apply(lambda x: 'Substance is Toxic' if x == 1 else 'Substance is not Toxic')
    results_df.drop(columns='prediction', inplace=True)
    return results_df


# define the inputs from the user
# class StringData(BaseModel):
#     #iupac_name: str
#     #common_name: str
#     #smiles: str
#     compound: str
#     target: str

# @app.post("/predict")
# def predict(data:StringData):
#     # Check if compound in test data
#     if data.compound == 'Please type in a compound!':
#         return data.compound

#     X_test = pd.read_csv('data/X_test.csv',index_col='ID')
#     X_predict = X_test[X_test.index==data.compound]

#     if X_predict.shape[0] == 0:
#         return 'Compound not found in Dataset'

#     # Check selected target and load model
#     if data.target == 'SR.MMP':
#         with open('production_model/fitted_xgb_mmp.pkl','rb') as f:
#             trained_model = pickle.load(f)
#     elif data.target == 'SR.ARE':
#         with open('production_model/fitted_xgb_are.pkl','rb') as f:
#             trained_model = pickle.load(f)
#     else:
#         return 'Please choose test: SR.MMP or SR.ARE'

#     # Predict and return
#     prediction = int(trained_model.predict(X_predict)[0])

#     if prediction == 1:
#         return {'Test type': data.target,
#                  data.compound: 'This compound is toxic'}

#     return {'Test type': data.target,
#             data.compound: 'This compound is not toxic'}

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
