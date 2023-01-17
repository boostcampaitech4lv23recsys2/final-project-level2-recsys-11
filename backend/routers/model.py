import pandas as pd
import numpy as np
import os
import pickle

from typing import List, Dict
from fastapi import APIRouter, Depends

from routers import data
from pydantic import BaseModel, Field

from datetime import datetime


class Model_Manager(BaseModel):
    model_name: str
    # hyperparam: str
    # item_vec: str
    # self.pred_item = pd.read_csv('/opt/ml/final-project-level2-recsys-11/RecBole/inference/EASE_a0b62c3f-295b-4295-911f-8da520bc4fc7.csv')
    created_at: datetime = Field(default_factory=datetime.now)

testing = Model_Manager(model_name='BPR')


NECESSARY_INFOS = ['ITEM_VECTOR', 'USER2IDX', 'ITEM2IDX', 'PRED_ITEM', 'PRED_SCORE']
# NECESSARY_INFOS = ['ITEM_VECTOR', 'USER2IDX', 'ITEM2IDX']


class ModelConfig:
    def __init__(self, config_path: str):
        self.model_name = None # model_name
        self.config_path = config_path
        self.necessary = {} # 필수적으로 들고있어야 할 정보들 (아이템 벡터, 아이디2idx, preds 등)
        self.hyper = {} # 각기 모델 config 파일에 있는 하이퍼 파람들 

        self.string_key = None

        # 지표 계산 클래스
        self.quantitative = None
        self.qualitative = None

    def load_config(self):
        from routers.metric import quantitative_indicator, qualitative_indicator

        with open(self.config_path, 'rb') as fr:
            infos = pickle.load(fr)
            for necessary_key in NECESSARY_INFOS:
                self.necessary[necessary_key] = infos[necessary_key]
                
            for hyper_k, hyper_v in infos.items():
                if hyper_k not in NECESSARY_INFOS:
                    self.hyper[hyper_k] = hyper_v

        self.quantitative = quantitative_indicator(data.dataset, infos['PRED_ITEM'], infos['PRED_SCORE'])
        self.qualitative = qualitative_indicator(data.dataset, infos['PRED_ITEM'], infos['PRED_SCORE'])

    def set_string_key(self, hyper_keys: list):
        # 하이퍼 파라미터 순으로 'uniform_5_0.0' 과 같은 키 생성
        # 여기서 하이퍼 파라미터 문자열 정렬은 필요 없을 수 있음
        # 어차피 파이썬 3.6 이상부터 dict 가 아이템 들어온 순서를 유지함.
        string_list = [str(self.hyper[hyper_key]) for hyper_key in hyper_keys]
        print(string_list)
        self.string_key = "_".join(string_list)


class ModelManager:
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.runs = {}
        self.model_name = None # 추가 필요

        self.hyper_keys = None # ['neg_sample_num', 'embedding_size' ... ]
        self._build_configs()

    def _build_configs(self):
        files = sorted(os.listdir(self.dir_path))
        
        for i, file in enumerate(files):
            config_path = os.path.join(self.dir_path, file)
            model_config = ModelConfig(config_path=config_path)
            model_config.load_config()

            if self.hyper_keys == None: # 최초 부른 모델 config 로 sanity check 기준 설정
                self.hyper_keys = list(model_config.hyper.keys())
            else:
                if set(self.hyper_keys) != set(model_config.hyper.keys()):
                    # 일치하지 않는 상태
                    continue
            
            model_config.set_string_key(self.hyper_keys)
            self.runs[model_config.string_key] = model_config

    def get_model_config(self, string_key: str) -> ModelConfig:
        return self.runs[string_key]
    
    def get_all_model_configs(self) -> List[ModelConfig]:
        return list(self.runs.values())

    def _sanity_check(self):
        pass


BPR_manager = ModelManager(dir_path='/opt/ml/final-project-level2-recsys-11/BPR_configs')
EASE_manager = ModelManager(dir_path='/opt/ml/final-project-level2-recsys-11/EASE_configs')

model_managers = {'BPR': BPR_manager, 'EASE': EASE_manager}
