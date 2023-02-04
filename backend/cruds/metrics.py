from typing import Dict
import pandas as pd
import numpy as np

from schemas.data import Dataset
from database.s3 import get_from_s3, s3_to_pd

class Quant_Metrics:
    '''
    Quantitative Metrics:
        - Recall_K: Dict
        - MAP_K: Dict
        - NDCG: Dict
        - AveragePopularity: Dict
        - Coverage: float
        - TailPercentage: float
    '''
    def __init__(self, dataset:Dataset, pred_item:Dict, K:int = 10):
        self.train_df = pd.DataFrame(dataset.train_df)
        self.ground_truth = pd.DataFrame(dataset.ground_truth)

        self.pred_item = pd.Series(pred_item.values(), index=[int(k) for k in pred_item.keys()], 
                                    name='item_id').apply(lambda x: x.astype(int))

        self.total_users = self.ground_truth.index

        self.n_user = dataset.n_user
        self.n_item = dataset.n_item 

        self.pop_per_item = dataset.popularity_per_item
        self.popularity_df = self.pred_item.apply(lambda R: [self.pop_per_item[int(item)] for item in R])
    
        self.K = K
    
    def Recall_K(self, users: None, predicted: None) -> Dict:
        if users == None:
            users = self.total_users

        if predicted == None:
            self.pred_item.loc[users]

        actual = self.ground_truth.loc[users, 'item_id']
        recall = {}

        for user in actual.index:
            act_set = set(actual[user].flatten())
            pred_set = set(predicted[user][:self.K].flatten())
            if len(act_set) != 0:
                recall[user] = len(act_set & pred_set) / min(len(act_set), len(pred_set))

        return recall
        
    def apk(self, actual, predicted, k) -> float:
        """
        Source:
        https://www.kaggle.com/code/nandeshwar/mean-average-precision-map-k-metric-explained-code/notebook

        Computes the average precision at k.
        This function computes the average precision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
                A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The average precision at k over the input lists
        """
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

    def MAP_K(self, users: None) -> Dict:
        """
        Computes the mean average precision at k.
        This function computes the mean average prescision at k between two lists
        of lists of items.
        """
        if users == None:
            users = self.total_users
                    
        actual = self.ground_truth.loc[users, 'item_id']
        predicted = self.pred_item.loc[users]

        return {user: self.apk(a, p, self.K) for user, a, p in zip(users, actual, predicted)}
        
    def NDCG(self, users: np.array) -> Dict:
        '''
        NDCG = DCG / IDCG
        DCG = rel(i) / log2(i + 1)
        '''
        if users == None:
            users = self.total_users
                    
        ndcg = {} 
        for user in users:
            k = min(self.K, len(self.ground_truth['item_id'].loc[1]))
            idcg = sum([1 / np.log2(j + 2) for j in range(k)]) # 최대 dcg. +2는 range가 0에서 시작해서
            dcg = sum([int(self.pred_item[user][j] in set(self.ground_truth['item_id'].loc[user])) 
                       / np.log2(j + 2) for j in range(self.K)])
            ndcg[user] = dcg / idcg

        return ndcg 

    def AveragePopularity(self, users:None) -> Dict:
        '''
        유저별로 주어진 추천 아이템 리스트의 평균 아이템 인기도 (self.popularity_df)
        -> 유저별 평균 추천 아이템 인기도 (popularity_metric)
        '''
        if users == None:
            users = self.total_users

        popularity_metric = self.popularity_df.loc[users].apply(lambda R: np.mean(R))

        return popularity_metric.to_dict()
    
    def Coverage(self) -> float:
        '''
        return: 추천된 아이템의 고유값 수 / 전체 아이템 수
        '''
        total_n_unique = set()
        for i in self.pred_item:
            total_n_unique |= set(i)

        return len(total_n_unique) / self.n_item
    
    def TailPercentage(self, tail_ratio=0.1) -> float:
        item_count = self.train_df.groupby('item_id').agg('count')
        item_count.drop(['rating', 'timestamp','origin_timestamp'], axis=1, inplace=True)
        item_count.columns = ['item_count']

        item_count_sort = item_count.sort_values(by = 'item_count', ascending=False)
        item_count_sort.reset_index(inplace=True)
        T = list(item_count_sort.item_id[-int(len(item_count_sort) * tail_ratio):].values)

        Tp = np.mean([sum([1 if item in T else 0 for item in self.pred_item[idx]]) / self.K
                            for idx in self.pred_item.index])
        return Tp
    
    def get_total_metrics(self, users: None) -> Dict:
        if users == None:
            users = self.total_users

        return {'Recall_K': self.Recall_K(users),
                'MAP_K': self.MAP_K(users),
                'NDCG': self.NDCG(users),
                'AveragePopularity': self.AveragePopularity(users)
                }


def avg_metric(rows: Dict) -> Dict:
    metrics = ['recall', 'ap', 'ndcg', 'tail_percentage', 'avg_popularity', \
                'diversity_cos', 'diversity_jac', 'serendipity_pmi', 'serendipity_jac', 'novelty'] # coverage not measured per user
    
    for row in rows: # row = experiment(dict)
        for key, value in row.items(): # key = column  # value = column value
            if key in metrics: # key = metric
                # get from s3 
                dict_of_per_user = get_from_s3(value) # {user1: (float),}
                row[key] = sum(dict_of_per_user.values()) / len(dict_of_per_user.values())
    
    return rows


async def predicted_per_item(pred_item_hash: str) -> pd.DataFrame:
    pred_item_pd = await s3_to_pd(pred_item_hash)
    pred_item_pd = pred_item_pd.reset_index()
    pred_item_pd.columns = ['rec_user', 'item_id']  ####### 나중에 수정
    predicted_per_item = pred_item_pd.explode('item_id').groupby('item_id').agg(list)
    return predicted_per_item
    