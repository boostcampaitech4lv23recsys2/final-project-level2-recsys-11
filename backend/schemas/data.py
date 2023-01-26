from pydantic import BaseModel
from typing import TypeVar, Dict
from functools import cached_property
import pandas as pd
import numpy as np
import numpy.typing as npt

class Dataset(BaseModel):
    dataset_name: str
    train_df: Dict
    ground_truth: Dict
    user_side: Dict
    item_side: Dict

    class Config:
        keep_untouched = ((cached_property,))

    @property
    def n_user(self):
        return self.train_df['user_id'].nunique()

    @property 
    def n_item(self):
        return self.train_df['item_id'].nunique()

    @cached_property
    def popularity_per_item(self):   # 유저 관점의 popularity - default
        '''
        popularity = (item과 상호작용한 유저 수) / (전체 유저 수) 

        return: dict('item_id': popularity, 'item_id2': popularity2, ...)
        '''
        pop_user_per_item = (self.train_df['item_id'].value_counts() / self.n_user).to_dict()

        return pop_user_per_item
    
    @cached_property
    def rating_matrix(self): # = implicit_matrix
        pass

    @cached_property
    def jaccard_matrix(self):
        pass
    
    @cached_property
    def pmi_matrix(self):
        pass 


class Experiment(BaseModel):
    experiment_id: str

    user_id: str
    dataset_name: str
    experiment_name: str
    alpha: float
    objective_fn: str

    hyperparameters: Dict
    pred_item: Dict
    pred_score: Dict
    item_vector: Dict
    
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