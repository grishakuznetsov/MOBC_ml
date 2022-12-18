import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
from sklearn.preprocessing import OneHotEncoder
import re
import numpy as np

app = FastAPI()

model = joblib.load('Ridge_cat.pkl')

df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
X_train_cat = df_train.drop(['name', 'selling_price'], axis=1)


def preprocess(x):
    if not type(x) == float:
        if re.search(r'\d*.?\d+', x):
            return float(re.search(r'\d*.?\d+', x).group())
        else:
            return np.nan
    else:
        return np.nan

def preprocess_torque(x):
    if not type(x) == float:
        matches = re.findall(r'\d*[.,]?\d+', x)
        max_torque = matches[-1].replace('.', '').replace(',', '')
        return float(matches[0]), float(max_torque)
    else:
        return np.nan


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
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


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item = dict(item)
    df = pd.DataFrame([item])

    enc = OneHotEncoder(handle_unknown='ignore')
    for i in ['fuel', 'seller_type', 'transmission', 'owner']:
        enc.fit(X_train_cat[[i]])
        encoder_df = pd.DataFrame(enc.transform(df[[i]]).toarray())
        encoder_df.columns = enc.get_feature_names_out()
        df = df.join(encoder_df)
    df = df.drop(['fuel', 'seller_type', 'transmission', 'owner'], axis=1)

    df['mileage'] = df['mileage'].apply(preprocess)
    df['engine'] = df['engine'].apply(preprocess)
    df['max_power'] = df['max_power'].apply(preprocess)
    split = pd.DataFrame(df['torque'].apply(preprocess_torque).to_list(),
                         columns=['torque', 'max_torque_rpm'])

    df = df.drop(['torque', 'selling_price'], axis=1)
    df = pd.concat([df, split], axis=1)

    df.columns = ['name', 'year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'torque', 'max_torque_rpm',
                  'fuel_CNG', 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol',
                  'seller_type_Dealer', 'seller_type_Individual',
                  'seller_type_Trustmark Dealer', 'transmission_Automatic',
                  'transmission_Manual', 'owner_First Owner',
                  'owner_Fourth & Above Owner', 'owner_Second Owner',
                  'owner_Test Drive Car', 'owner_Third Owner']

    df = df.drop(['name'], axis=1)

    print(df.columns)
    return model.predict(df)[0]




@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    dict_list = []
    for i in items:
        dict_list.append(dict(i))

    df = pd.DataFrame(dict_list)

    enc = OneHotEncoder(handle_unknown='ignore')
    for i in ['fuel', 'seller_type', 'transmission', 'owner']:
        enc.fit(X_train_cat[[i]])
        encoder_df = pd.DataFrame(enc.transform(df[[i]]).toarray())
        encoder_df.columns = enc.get_feature_names_out()
        df = df.join(encoder_df)
    df = df.drop(['fuel', 'seller_type', 'transmission', 'owner'], axis=1)

    df['mileage'] = df['mileage'].apply(preprocess)
    df['engine'] = df['engine'].apply(preprocess)
    df['max_power'] = df['max_power'].apply(preprocess)
    split = pd.DataFrame(df['torque'].apply(preprocess_torque).to_list(),
                         columns=['torque', 'max_torque_rpm'])

    df = df.drop(['torque', 'selling_price'], axis=1)
    df = pd.concat([df, split], axis=1)

    df.columns = ['name', 'year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'torque', 'max_torque_rpm',
                  'fuel_CNG', 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol',
                  'seller_type_Dealer', 'seller_type_Individual',
                  'seller_type_Trustmark Dealer', 'transmission_Automatic',
                  'transmission_Manual', 'owner_First Owner',
                  'owner_Fourth & Above Owner', 'owner_Second Owner',
                  'owner_Test Drive Car', 'owner_Third Owner']

    df = df.drop(['name'], axis=1)

    print(df.columns)
    return list(model.predict(df))