from pydantic import BaseModel
from typing import TypeVar, Dict
import pandas as pd
import numpy as np
import numpy.typing as npt

# pd.Dataframe -> 테스트 필요 
pd_DataFrame = TypeVar('pandas.core.frame.DataFrame')

class Dataset(BaseModel):
    dataset_name: str
    train_df: pd_DataFrame
    ground_truth: pd_DataFrame
    user_side: pd_DataFrame
    item_side: pd_DataFrame


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
    item_vector: np.array
    
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