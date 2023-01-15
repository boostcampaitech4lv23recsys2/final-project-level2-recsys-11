import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends

from data import dataset_info
import dependencies 
from datetime import datetime

router = APIRouter()

@router.get('/K')
def get_dataset_info(dataset_info=Depends(dependencies.get_dataset)):
    response = {'K': dataset_info.K}
    return response

class Model_Manager:
    def __init__(self):
        self.model_name = 'BPR'
        self.hyperparam = ['embedding_size', 'negative_sample', 'weight_decay']
        self.item_vec = np.array([[1,2,3],[4,5,6],[7,8,9]])
        # self.pred_item = pd.read_csv('/opt/ml/final-project-level2-recsys-11/RecBole/inference/EASE_a0b62c3f-295b-4295-911f-8da520bc4fc7.csv')
        self.datetime = datetime.now()

BPR_Manager = Model_Manager()