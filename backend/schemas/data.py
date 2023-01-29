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
    upload_time: datetime = Field(default_factory=datetime.now)

    train_df: Dict
    ground_truth: Dict
    user_side_df: Dict
    item_side_df: Dict
    desc: str='' 

    item_popularity: Dict 
    item_name: Dict 
    item_vectors: Dict[str, Dict[str, List]] 

    
    @property
    def n_user(self):
        return self.train_df['user_id'].nunique()

    @property 
    def n_item(self):
        return self.train_df['item_id'].nunique()
    

class Experiment(BaseModel):
    ID: str
    dataset_name: str
    experiment_name: str
    alpha: float = 1.0
    objective_fn: str = None

    hyperparameters: Dict
    pred_item: Dict
    pred_score: Dict
    item_vector: Dict
    distance_matrix: Dict
    jaccard_matrice: Union[Dict, None] 
    
    recall: Dict
    map: Dict
    ndcg: Dict
    tail_per: Dict
    avg_pop: Dict
    coverage: Dict

    diversity_cos: Dict
    diversity_jac: Dict
    serendipity_pmi: Dict
    serendipity_jac: Dict
    novelty: Dict 
