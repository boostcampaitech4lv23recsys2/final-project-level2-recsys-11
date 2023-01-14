import pandas as pd
import numpy as np
import os
import pickle

from typing import List, Tuple

NECESSARY_INFOS = ['ITEM_VECTOR', 'USER2IDX', 'ITEM2IDX', 'TOPK', 'SCORE']


class ModelConfig:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.necessary = {} # 필수적으로 들고있어야 할 정보들 (아이템 벡터, 아이디2idx, preds 등)
        self.hyper = {} # 각기 모델 config 파일에 있는 하이퍼 파람들 

        self.string_key = None


    def load_config(self):
        with open(self.config_path, 'rb') as fr:
            infos = pickle.load(fr)
            for necessary_key in NECESSARY_INFOS:
                self.necessary[necessary_key] = infos[necessary_key]
                
            for hyper_k, hyper_v in infos.items():
                if hyper_k not in NECESSARY_INFOS:
                    self.hyper[hyper_k] = hyper_v


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
        self.models = {}

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
            self.models[model_config.string_key] = model_config

    
    def get_model_config(self, string_key: str) -> ModelConfig:
        return self.models[string_key]

    
    def get_all_model_configs(self) -> List[ModelConfig]:
        return list(self.models.values())


    def _sanity_check(self):
        pass
            


if __name__ == '__main__':
    BPR_manager = ModelManager(dir_path='/opt/ml/final-project-level2-recsys-11/EASE_configs')
    
    print(BPR_manager.models.keys())
    b = BPR_manager.get_model_config('100.0') # 스트림릿에서 이런식으로 호출 가능함.
    print(b.necessary['ITEM_VECTOR'].shape) # EASE 는 similarity 인데 item_vector 가 아닌데? 이미 similarity 가 계산되어 있다고 봐도 무방
    print(b.hyper['reg_weight'])
    print(BPR_manager.get_all_model_configs()[0].hyper)