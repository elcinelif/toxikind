from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
from pydantic import BaseModel

MODEL_INFO_DICT = {
    'model_GBC_ahr': {
        'abbreviation': 'NR-AhR',
        'full_name': 'Aryl Hydrocarbon (NR-AhR)',
        'threshold': 0.5
    },
    'model_GBC_ar': {
        'abbreviation': 'NR-AR',
        'full_name': 'Androgen Receptor (NR-AR)',
        'threshold': 0.5
    },
    'model_GBC_are': {
        'abbreviation': 'SR-ARE',
        'full_name': 'Antioxidant Response (SR-ARE)',
        'threshold': 0.5
    },
    'model_GBC_aromatase': {
        'abbreviation': 'NR-Arom',
        'full_name': 'Aromatase Inhibition (NR-Arom)',
        'threshold': 0.5
    },
    'model_GBC_atad5': {
        'abbreviation': 'SR-ATAD5',
        'full_name': 'DNA Damage (SR-ATAD5)',
        'threshold': 0.5
    },
    'model_GBC_er_lbd': {
        'abbreviation': 'NR-ER-LBD',
        'full_name': 'Estrogen Receptor LBD (NR-ER-LBD)',
        'threshold': 0.5
    },
    'model_GBC_er': {
        'abbreviation': 'NR-ER',
        'full_name': 'Estrogen Receptor (NR-ER)',
        'threshold': 0.5
    },
    'model_GBC_hse': {
        'abbreviation': 'SR-HSE',
        'full_name': 'Heat Shock Response (SR-HSE)',
        'threshold': 0.5
    },
    'model_GBC_lbd': {
        'abbreviation': 'NR-AR-LBD',
        'full_name': 'Androgen Receptor LBD (NR-AR-LBD)',
        'threshold': 0.5
    },
    'model_GBC_mmp': {
        'abbreviation': 'SR-MMP',
        'full_name': 'Mitochondrial Membrane (SR-MMP)',
        'threshold': 0.5
    },
    'model_GBC_p53': {
        'abbreviation': 'SR-p53',
        'full_name': 'p53 Stress Response (SR-p53)',
        'threshold': 0.5
    },
    'model_GBC_ppar_gamma': {
        'abbreviation': 'NR-PPARγ',
        'full_name': 'PPAR-gamma (NR-PPARγ)',
        'threshold': 0.5
    }
}

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

    for model in MODEL_INFO_DICT.keys():

        with open(f'production_model/{model}.pkl','rb') as f:
            trained_model= pickle.load(f)

        prediction_probability = trained_model.predict_proba(X_predict)[:,-1][0]
        MODEL_INFO_DICT[model]['toxicity_probability'] = round(prediction_probability,4)
        threshold = MODEL_INFO_DICT[model]['threshold']
        if prediction_probability > threshold:
            MODEL_INFO_DICT[model]['Toxic?'] = 'Yes'
        else:
            MODEL_INFO_DICT[model]['Toxic?'] = 'No'

    results_df = pd.DataFrame(MODEL_INFO_DICT).transpose().reset_index()
    results_df.drop(columns=['threshold','index'], inplace=True)
    results_df.index += 1
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
