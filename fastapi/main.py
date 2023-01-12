from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.io as pio
import os 
import json

app = FastAPI()
df = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')[:10]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/data")
def road_data():

    return df.to_dict(orient='records')


@app.get("/plot/")
async def create_plot():
    trace = go.Scatter(x=[1, 2, 3], y=[4, 5, 6])
    data = [trace]
    layout = go.Layout(title="My Plot")
    fig = go.Figure(data=data, layout=layout)

    return pio.to_json(fig)