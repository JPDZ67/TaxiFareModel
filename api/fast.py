from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from typing import Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict_fare")
def predict_fare(pickup_datetime, pickup_longitude:float, pickup_latitude:float, dropoff_longitude:float, dropoff_latitude:float, passenger_count)->Dict[str,float]:

    params_ = {"key":["2013-07-06 17:18:00.000000000"],
            "pickup_datetime": [pickup_datetime],
            "pickup_longitude": [pickup_longitude],
            "pickup_latitude": [pickup_latitude],
            "dropoff_longitude": [dropoff_longitude],
            "dropoff_latitude": [dropoff_latitude],
            "passenger_count": [int(passenger_count)]
            }
    
    df_ = pd.DataFrame(params_)
    
    pipeline = joblib.load("XGBoost_finalized_model.pkl")

    predict_ = pipeline.predict(df_)

    return {"prediction":float(predict_[0])}

    