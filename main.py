# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import sys

class DataIngest():
    @staticmethod
    def data_ingest():
        """
        Perform data ingestion from remote source.
        """
        # 162 for forest fires
        dataset_ = fetch_ucirepo(id=162)
        X = dataset_.data.features
        y = dataset_.data.targets
        return pd.concat([X, y], axis = 1)
    
if __name__ == '__main__':
    output_file = sys.argv[1]
    data = DataIngest.data_ingest()
    data.to_csv(output_file, index=False)


# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the target names (class labels)
data = DataIngest()
target_names = data.data_ingest

# Define the input data format for prediction
class ForestFire(BaseModel):
    features: List[float]

# Initialize FastAPI
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
def predict(ff_data: ForestFire):
    if len(ff_data.features) != model.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Input must contain {model.n_features_in_} features."
        )
    # Predict
    prediction = model.predict([ff_data.features])[0]
    prediction_name = target_names[prediction]
    return {"prediction": int(prediction), "prediction_name": prediction_name}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Fire Forest model API"}