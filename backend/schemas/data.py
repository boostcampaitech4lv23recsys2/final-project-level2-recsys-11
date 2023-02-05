from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Union, List
from functools import cached_property
import pandas as pd
import numpy as np
import numpy.typing as npt

class Dataset(BaseModel):
    ID: str
    dataset_name: str
    upload_time: Union[str, None]

    train_interaction: Dict
    ground_truth: Dict
    user_side: Dict
    item_side: Dict
    dataset_desc: str

    # item_vectors: Dict[str, Dict[str, List]] 

    @property
    def n_user(self):
        return self.train_interaction['user_id'].nunique()

    @property 
    def n_item(self):
        return self.train_interaction['item_id'].nunique()


# class TrainInteraction(BaseModel):
#     user_id: Union[str, int]
#     item_id: Union[str, int]
#     score: Union[int, float, None]
#     timestamp: Union[int, None]


# class GroundTruth(BaseModel):
#     user_id: Union[str, int]
#     item_id: Union[str, int]
    

class CoreDataset(BaseModel):
    dataset_name: str
    train_interaction: Dict
    ground_truth: Dict


class Experiment(BaseModel):
    ID: str
    dataset_name: str
    experiment_name: str
    alpha: float = 1.0
    objective_fn: Union[str, None]

    hyperparameters: str
    pred_items: Dict # 디코
    pred_scores: Union[Dict, None] # 디코
    # cos_dist: Union[Dict, None] # 디코
    # pmi_dist: Union[Dict, None] # 디코
    # jac_dist: Union[Dict, None] # 디코

    user_tsne: Union[Dict, None]
    item_tsne: Union[Dict, None]
    
    recall: float
    ndcg: float
    map: float
    avg_popularity: float
    tail_percentage: float
    coverage: float

    diversity_cos: float
    serendipity_pmi: float
    novelty: float

    diversity_jac: Union[float, None]
    serendipity_jac: Union[float, None]

    metric_per_user: Dict # 그대로
