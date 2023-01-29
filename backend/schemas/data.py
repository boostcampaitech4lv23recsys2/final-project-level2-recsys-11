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
    train_df: Dict
    ground_truth: Dict
    user_side_df: Dict
    item_side_df: Dict
    # user2idx: Dict
    # item2idx: Dict
    desc: str='' 

    popularity_per_item: Dict # {item_id: popularity (float)}
    item_name: Dict # {item_id: (movie_title)}
    item_vectors: Dict[Dict[str, List]] # 장르스 같은 자카드

    upload_time: datetime = Field(default_factory=datetime.now())
    
    @property
    def n_user(self):
        return self.train_df['user_id'].nunique()

    @property 
    def n_item(self):
        return self.train_df['item_id'].nunique()
    

class Experiment(BaseModel):
    experiment_id: str

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
