from pydantic import BaseModel
from typing import TypeVar, Dict

import numpy as np
import numpy.typing as npt

# pd.Dataframe -> 테스트 필요 
pd_DataFrame = TypeVar('pandas.core.frame.DataFrame')

class Dataset(BaseModel):
    train_df: pd_DataFrame
    ground_truth: pd_DataFrame
    user_side: pd_DataFrame
    item_side: pd_DataFrame
