from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from toxikind.main import *

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
def predict(compound):
    """
    Make a single compound prediction
    - Take input from interface
    - Process
    - Call model
    - Predict toxicity
    """
    pass

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
