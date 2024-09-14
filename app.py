from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Dict
import os

# Define the FastAPI app
app = FastAPI()

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load your model from the same directory as the app.py file
model_path = os.path.join(current_dir, 'model.pkl')
model = joblib.load(model_path)

# Define a request body model using Pydantic
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API!"}

@app.post("/predict")
def predict(features: IrisFeatures):
    # Convert the input features to a DataFrame
    input_data = pd.DataFrame([features.dict()])

    # Make prediction
    prediction = model.predict(input_data)

    # Return the result as a JSON response
    return {"prediction": prediction.tolist()}
