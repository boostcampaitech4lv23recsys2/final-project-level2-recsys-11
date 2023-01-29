import pandas as pd
import numpy as np

from typing import Dict

class W4RExperiment:
    __hyper_parameters: Dict
    __prediction_matrix: pd.DataFrame

    def upload_experiment(
        hyper_parameters: Dict,
        prediction_matrix: pd.DataFrame
    ) -> None:

    W4RExperiment.__hyper_parameters = hyper_parameters


