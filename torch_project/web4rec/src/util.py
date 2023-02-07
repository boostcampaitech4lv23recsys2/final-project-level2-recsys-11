import os
import pickle
from typing import Union, List, Dict

import requests
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, linalg

import json

class RoundingFloat(float):
    __repr__ = staticmethod(lambda x: format(x, '.6f'))

json.encoder.c_make_encoder = None
if hasattr(json.encoder, 'FLOAT_REPR'):
    # Python 2
    json.encoder.FLOAT_REPR = RoundingFloat.__repr__
else:
    # Python 3
    json.encoder.float = RoundingFloat


class W4RException(Exception):
    pass


class Web4RecDataset:
    def __init__(self, dataset_name: str):
        self.dataset_name: str = dataset_name

        self.dataset_desc: str = dataset_name # desc 가 없으면 이름

        self.train_interaction: pd.DataFrame = None

        self.user_ids = None
        self.item_ids = None

        self.n_users: int = None
        self.n_items: int = None

        self.ground_truth: pd.DataFrame = None
        self.user_side: pd.DataFrame = None
        self.item_side: pd.DataFrame = None


    def add_train_interaction(self, train_interaction: pd.DataFrame):
        # string key
        train_interaction = train_interaction.astype({'user_id': str,'item_id': str})

        # train_interaction['user_id'] = train_interaction['user_id'].astype(str)
        # train_interaction['item_id'] = train_interaction['item_id'].astype(str)

        self.user_ids = train_interaction['user_id'].unique().tolist()
        self.item_ids = train_interaction['item_id'].unique().tolist()

        # self.n_users, self.n_items
        self.n_users = train_interaction['user_id'].nunique()
        self.n_items = train_interaction['item_id'].nunique()

        # self.train_interaction
        self.train_interaction = train_interaction[['user_id', 'item_id']]

    
    def add_ground_truth(self, ground_truth):
        if self.train_interaction is None:
            raise W4RException('ground_truth: You should add train interaction first.')
        
        # train_interaction = train_interaction.copy()
        ground_truth = ground_truth.astype({'user_id': str,'item_id': str})

        ground_truth_user_ids = ground_truth['user_id'].unique().tolist()
        ground_truth_item_ids = ground_truth['item_id'].unique().tolist()

        extra_users = set(ground_truth_user_ids) - set(self.user_ids)
        if extra_users:
            pass
            # print(f'ground_truth: {list(extra_users)[:5]}... users are not in train_interaction.')

        extra_items = set(ground_truth_item_ids) - set(self.item_ids)
        if extra_items:
            pass
            # print(f'ground_truth: {list(extra_items)[:5]}... items are not in train_interaction.')

        self.ground_truth = ground_truth

    
    def add_user_side(self, user_side: pd.DataFrame):
        """
        Should put only cate col.
        pandas category not support
        """
        if self.train_interaction is None:
            raise W4RException('user_side: You should add train interaction first.')
        if self.ground_truth is None:
            raise W4RException('user_side: You should add ground truth first.')

        user_side = user_side.astype({'user_id': str})

        # 그냥 train user 에 없는 유저는 뺀다.
        user_side_user_ids = user_side['user_id'].unique().tolist()
        extra_users = set(user_side_user_ids) - set(self.user_ids)
        if extra_users:
            pass
            # print(f'user_side: {list(extra_users)[:5]}... users are not in train_interaction. they will be dropped.')
        user_side = user_side[user_side['user_id'].isin(self.user_ids)]

        # user_side 를 돌면서 str, int 로만 구성되는지 체크함.
        if user_side.columns.tolist().remove('user_id'):
            raise W4RException(f'user_side: there is no user_side information. (only user_id)')

        for user_col in user_side.columns:
            if user_side[user_col].dtype == float:
                raise W4RException(f'user_side: {user_col} column dtype is not str or int.')

        self.user_side = user_side


    def add_item_side(self, item_side: pd.DataFrame):
        """
        Should put only cate col.
        pandas category not support
        item_id, item_name null, item_vector null
        """
        if self.train_interaction is None:
            raise W4RException('item_side: You should add train interaction first.')
        if self.ground_truth is None:
            raise W4RException('item_side: You should add ground truth first.')

        item_side = item_side.astype({'item_id': str})

        item_side_item_ids = item_side['item_id'].unique().tolist()
        extra_items = set(item_side_item_ids) - set(self.item_ids)
        if extra_items:
            pass
            # print(f'item_side: {list(extra_items)[:5]} items are not in train_interaction. they will be dropped.')
        item_side = item_side[item_side['item_id'].isin(self.item_ids)]

        # item_side 를 돌면서 str, int 로만 구성되는지 체크함.
        if item_side.columns.tolist().remove('item_id'):
            raise W4RException(f'item_side: there is no item_side information. (only item_id)')

        for item_col in item_side.columns:
            if item_side[item_col].dtype == float:
                raise W4RException(f'item_side: {item_col} column dtype is not str or int.')

        # popularity 구해주기
        train = self.train_interaction
        item_popularity = \
            train.groupby('item_id')['user_id'].count() / train['user_id'].nunique()
        item_side['item_popularity'] = item_popularity.values

        # ':multi' 가 있다면 처리해주기
        multi_col_name = None
        for col in item_side.columns:
            if ':multi' in col:
                multi_col_name = col
                break
        
        # item_vector 생성
        if multi_col_name is not None:
            multis = item_side[multi_col_name].apply(lambda x: x.split(' '))
            multis.index = item_side['item_id']
            multi_set = set()
            for _, elems in multis.iteritems():
                multi_set = multi_set | set(elems)

            elem2idx = {v:k for k, v in enumerate(multi_set)}
            multis = multis.apply(lambda multi: [elem2idx[elem] for elem in multi])

            for item_id, multi_hot in multis.iteritems():
                vector = np.zeros(len(multi_set), dtype=int)
                vector[multi_hot] = 1
                multis.loc[item_id] = vector.tolist()

            item_side['item_vector'] = item_side['item_id'].map(multis)
        
        self.item_side = item_side

    