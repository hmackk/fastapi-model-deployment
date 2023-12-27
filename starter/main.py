import pickle
from typing import Optional, Union

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()


def to_hyphen(string: str) -> str:
    return string.replace("_", "-")


class Data(BaseModel):
    age: Optional[Union[int, list]] = 35
    workclass: Optional[Union[str, list]] = "Self-emp-inc"
    fnlgt: Optional[Union[int, list]] = 76515
    education: Optional[Union[str, list]] = "Bachelors"
    education_num: Optional[Union[int, list]] = 10
    marital_status: Optional[Union[str, list]] = "Never-married"
    occupation: Optional[Union[str, list]] = "Exec-managerial"
    relationship: Optional[Union[str, list]] = "Not-in-family"
    race: Optional[Union[str, list]] = "Black"
    sex: Optional[Union[str, list]] = "Female"
    capital_gain: Optional[Union[int, list]] = 6114
    capital_loss: Optional[Union[int, list]] = 0
    hours_per_week: Optional[Union[int, list]] = 40
    native_country: Optional[Union[str, list]] = "Italy"

    class Config:
        alias_generator = to_hyphen


@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    model = pickle.load(open("model/trained_model.pkl", "rb"))
    encoder = pickle.load(open("model/encoder.pkl", "rb"))
    lb = pickle.load(open("model/labelizer.pkl", "rb"))


@app.get("/")
def read_root():
    """
    Root endpoint for a welcome message.
    """
    return {"message": "Hello! This is a welcome message!"}


@app.post("/predict")
def predict(input: Data):
    model = pickle.load(open("starter/model/trained_model.pkl", "rb"))
    encoder = pickle.load(open("starter/model/encoder.pkl", "rb"))
    lb = pickle.load(open("starter/model/labelizer.pkl", "rb"))
    df = pd.DataFrame([input.dict(by_alias=True)])
    X, _, _, _ = process_data(
        df,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X)
    output = lb.inverse_transform(preds)
    return {"predictions": str(output)}
