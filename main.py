from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import pickle
import io

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

    def to_dataframe(self):
        dict_item = self.model_dump()
        df = pd.DataFrame(data=[list(dict_item.values())], columns=list(dict_item.keys()))
        return df



class Items(BaseModel):
    objects: List[Item]

def preproccessing(df: pd.DataFrame):
    df = df.drop(columns=['name','fuel', 'seller_type', 'transmission', 'owner', 'torque'])
    df['mileage'] = df['mileage'].agg(lambda x: str(x)[:next((i for i, c in enumerate(str(x)) if c.isalpha()), -1)])
    df['engine'] = df['engine'].agg(lambda x: str(x)[:next((i for i, c in enumerate(str(x)) if c.isalpha()), -1)])
    df['max_power'] = df['max_power'].agg(lambda x: str(x)[:next((i for i, c in enumerate(str(x)) if c.isalpha()), -1)])
    df = df.astype({'mileage':'float', 'engine':'int', 'max_power':'float', 'seats':'int'})
    with open("scaler.pkl", 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    df = scaler.transform(df)
    return df

def make_prediction(df: pd.DataFrame):
    with open("model.pkl", 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    res = loaded_model.predict(df)[0]
    return res


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    test = preproccessing(item.to_dataframe())
    result = make_prediction(test)
    return result


@app.post("/predict_items")
def predict_items(items: UploadFile):
    content = items.file.read()
    df = pd.read_csv(io.BytesIO(content))
    test = preproccessing(df)
    df['selling_price'] = make_prediction(test)
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()