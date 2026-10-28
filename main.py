from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.joblib")

class HouseFeatures(BaseModel):
    GrLivArea: float
    BedroomAbvGr: int
    FullBath: int
    YearBuilt: int

@app.post("/predict")
def predict_price(features: HouseFeatures):
    data = pd.DataFrame([features.model_dump()])
    prediction = model.predict(data)
    return {"predicted_price": prediction[0]}