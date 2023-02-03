
import requests
import pandas as pd
import numpy as np
from scipy import sparse


class W4RException(Exception):
    pass


class W4RExperiment:
    __server_url: str = 'http://127.0.0.1:8000/web4rec-lib' 
    __ID: str = None # user_id -> ID #
    __password: str = None

    dataset_name: str = None
    experiment_name: str = None
    hyper_parameters: dict = None


    def login(ID: str, password: str):
            response = requests.get(
                url=W4RExperiment.__server_url + '/login',
                params={
                    'ID': ID,
                    'password': password
                }
            ).json()

            if response[0] == ID:
                print(f'login: {response[0]} log in success.')
            else:
                raise W4RException(f'login: ID[{ID}] failed.')

            W4RExperiment.__ID = ID
            W4RExperiment.__password = password

    
    def upload_experiment(
        dataset_name: str,
        experiment_name: str,
        hyper_parameters: dict,
        prediction_matrix: pd.DataFrame
    ):
        W4RExperiment.dataset_name = dataset_name
        W4RExperiment.experiment_name = experiment_name
        W4RExperiment.hyper_parameters = hyper_parameters

        # 유저아이디+데이터셋 이름 의 코어 파일이 있는지 찾는다.
        # 있다면 불러오고, 없으면
        # get 메소드로 전달받는다.
        # 전달받으면 유저아이디+데이터셋이름 코어파일 저장한다.

        # 코어파일에서 필요한 것.
        # ground_truth 필요
        # train_interaction 필요
        # item_vectors 필요



