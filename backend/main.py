from typing import Dict, TypeVar
import numpy as np
import numpy.typing as npt


from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.io as pio
import os 
# import json

from routers import data, metric, model
import dependencies


app = FastAPI()
# df = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')[:10]

app.include_router(model.router, prefix='/model')
app.include_router(metric.router, prefix='/metric')


@app.get("/")
def read_root():
    return {"Hello": "World"}

# @app.get("/data")
# def road_data():

#     return df.to_dict(orient='records')

# @app.get('/dataset/{attribute}')
# def get_dataset(attribute, dataset=Depends(dependencies.get_dataset)):
#     return {"From dataset_info: ": data.dataset.attribute}

# @app.get('/model_manager')
# def get_model_manager(model_manager=Depends(dependencies.get_model_manager)):
#     return {'model_name': str(model_manager.datetime)}


# @app.get("/data/{data_path}", description = 'data_path 이름에 있는 데이터들을 가져')
# def load_data(data_path: str, desc):
#     path = os.path.join("/opt/ml/final-project-level2-recsys-11/dataset", )
#     df = pd.read_csv(path)

#     return df.to_dict(orient='records')

 
@app.get("/plot/")
async def create_plot():
    trace = go.Scatter(x=[1, 2, 3], y=[4, 5, 6])
    data = [trace]
    layout = go.Layout(title="My Plot")
    fig = go.Figure(data=data, layout=layout)

    return pio.to_json(fig)

# 클래스
# TODO: class dataset, model_run, model_manager

# 데이터 업로드/올리기 함수 
# TODO: upload_data 
# TODO: config, 

# 불러오기 함수
# TODO: get_model_run, 


# class dataset(BaseModel):
#     train_df: pd_DataFrame
#     ground_truth: pd_DataFrame


# class Model_Run(BaseModel):
#     model_name: str
#     run_name: str
#     hyper_param: Dict[str, float] = Field(default_factory=dict)
#     pred_score: npt.NDArray
#     pred_items: npt.NDArray
#     item_vector: npt.NDArray

#     def load_infos(self):
#         pass
    
# model_managers = dict()

# class Model_Manager(BaseModel):
#     model_name: str
#     runs: Dict[str, Model_Run] = Field(default_factory=dict)

#     # created_at: datetime = Field(default_factory=datetime.now)
#     # updated_at: datetime = Field(default_factory=datetime.now)

#     def add_run(self, model_run: Model_Run):
#         # 



# # streamlit에서 데이터를 올릴때 -- 지금은 path로 access 
# @app.post("/model_run", description='model_run 업로드')


