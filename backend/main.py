from typing import Dict, TypeVar
import numpy as np
import numpy.typing as npt
import uvicorn
from collections import defaultdict

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.io as pio
import os 

from routers import data, metric, model

app = FastAPI()

app.include_router(metric.router, prefix='/metric')


@app.get("/")
def read_root():
    return {"Hello": "World"}
    
 
@app.get("/plot/")
async def create_plot():
    trace = go.Scatter(x=[1, 2, 3], y=[4, 5, 6])
    data = [trace]
    layout = go.Layout(title="My Plot")
    fig = go.Figure(data=data, layout=layout)

    return pio.to_json(fig)

@app.get('/model_hype_type', description='모델의 하이퍼파라미터 종류를 가져옵니다.')
def model_hype_type() -> dict:
    result = dict()
    for key in model.model_managers.keys():
        result[key] = model.model_managers[key].possible_hyper_param
    return result

@app.get('/cal_metric', description='보내온 실험에 대해 metric을 계산합니다.')
def cal_metric(model_name:str, str_key:str) -> dict:
    result = dict()
    result[model_name] = model_name
    result[str_key] = str_key
    return result
# 클래스
# TODO: class dataset, model_run, model_manager

# 데이터 업로드/올리기 함수 
# TODO: upload_data 
# TODO: config, 

# 불러오기 함수
# TODO: get_model_run, 




if __name__ == '__main__':
    uvicorn.run(app)