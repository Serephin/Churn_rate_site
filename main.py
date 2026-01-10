import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
import json
import re
import pickle
from typing import Optional
#uvicorn main:app --reload --host 0.0.0.0 --port 8000
app=FastAPI()

@app.get("/") #decorator for  defining app path for method 'GET' in our app which returns this text

def root():
    return {"what does this app do?":"Predicts whether the person will leave the bank"}


class ChurnInput(BaseModel):
    ID:int
    ID_клиента: float
    фамилия: str
    кредитный_рейтинг: float
    город: str
    пол: str
    возраст: float
    стаж_в_банке: float
    баланс_депозита: Optional[float] = None
    число_продуктов: int
    есть_кредитка: float   
    активный_клиент: float 
    оценочная_зарплата: float
    

def json_to_df(js):
    if isinstance(js, dict):
        dataFrame = pd.DataFrame([js])
    elif isinstance(js, list) and all(isinstance(x, dict) for x in js):
        dataFrame = pd.DataFrame(js)
    else:
        raise ValueError(f"Неподдерживаемый формат: {type(js)}")
    return dataFrame

def transform_categorical(df):
    enc = joblib.load('encoder.pkl')
    encoded=enc.transform(df[['город','пол']]).toarray()
    encoded=pd.DataFrame(encoded, columns=enc.get_feature_names_out(['город','пол']))
    df=pd.concat([df, encoded], axis=1)
    return df


def balance_preprocess(df):
    df['balance_null']=df.баланс_депозита.isna()
    df.баланс_депозита.fillna(df.баланс_депозита.median(), inplace=True)
    df['balance_null']=df['balance_null'].astype(int)
    return df

def scaling(df):
    scaler=load('std_scaler.bin')
    cols_to_scale=['кредитный_рейтинг', 'возраст', 'стаж_в_банке', 'баланс_депозита',
       'оценочная_зарплата']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df
    
def make_prediction(df):
    lgbm = joblib.load('lgb.pkl')
    pred=lgbm.predict(df)
    return pred


@app.post("/predict")
def predict_endpoint(data: ChurnInput):
    df = json_to_df(data.dict())           
    df = transform_categorical(df)
    df.drop(['город','фамилия','пол', 'ID_клиента', 'ID'], axis=1, inplace=True)
    df = balance_preprocess(df)
    df = scaling(df)
    
    pred = make_prediction(df)            
    pred_list = pred.tolist()             
    
    return {
        "prediction": pred_list,
        "input": data.dict()              
    }
