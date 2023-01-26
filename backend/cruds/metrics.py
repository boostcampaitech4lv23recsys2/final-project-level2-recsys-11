from typing import Dict
from pandas import pd
from numpy import np

from schemas.data import Dataset

class Quant_Metrics:
    '''
    Quantitative Metrics:
        - Recall_K
        - MAP_K
        - NDCG
        - Coverage
        - TailPercentage
        - AveragePopularity 
    '''
    def __init__(self, dataset:Dataset, pred_item:Dict, pred_score:Dict, K:int = 10):
        self.train_df = pd.DataFrame(dataset.train_df)
        self.ground_truth = pd.DataFrame(dataset.ground_truth)

        self.pred_item = pd.Series(pred_item.values(), index=[int(k) for k in pred_item.keys()], 
                                    name='item_id').apply(lambda x: x.astype(int))
        self.pred_score = pd.Series(pred_score.values(), index=[int(k) for k in pred_score.keys()], 
                                    name='item_id') 

        self.n_user = dataset.n_user
        self.n_item = dataset.n_item 

        self.pop_per_item = dataset.popularity_per_item
        self.popularity_df = self.pred_item.apply(lambda R: [self.pop_per_item[int(item)] for item in R])
    
        self.K = K
    
    def Recall_K(self) -> float:
        actual = self.ground_truth['item_id']
        predicted = self.pred_item
        sum_recall = 0.0; true_users = 0

        for idx in actual.index:
            act_set = set(actual[idx].flatten())
            pred_set = set(predicted[idx][:self.K].flatten())
            if len(act_set) != 0:
                sum_recall += len(act_set & pred_set) / min(len(act_set), len(pred_set))
                true_users += 1

        return sum_recall / true_users
    
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

    def MAP_K(self) -> float:
        """
        Computes the mean average precision at k.
        This function computes the mean average prescision at k between two lists
        of lists of items.
        """
        return np.mean([self.apk(a, p, self.K) for a, p in zip(self.ground_truth.values, self.pred_item.values)])

    def NDCG(self) -> float:
        '''
        NDCG = DCG / IDCG
        DCG = rel(i) / log2(i + 1)
        '''
        ndcg = 0
        for i in self.pred_item.index:
            k = min(self.K, len(self.ground_truth['item_id'].loc[1]))
            idcg = sum([1 / np.log2(j + 2) for j in range(k)]) # 최대 dcg. +2는 range가 0에서 시작해서
            dcg = sum([int(self.pred_item[i][j] in set(self.ground_truth['item_id'].loc[i])) / np.log2(j + 2) for j in range(self.K)])
            ndcg += dcg / idcg

        return ndcg / len(self.pred_item)
    
    def Coverage(self) -> float:
        '''
        return: 추천된 아이템의 고유값 수 / 전체 아이템 수
        '''
        total_n_unique = set()
        for i in self.pred_item:
            total_n_unique |= set(i)

        return len(total_n_unique) / self.n_item
    
    def TailPercentage(self, tail_ratio=0.1):
        item_count = self.train_df.groupby('item_id').agg('count')
        item_count.drop(['rating', 'timestamp','origin_timestamp'], axis=1, inplace=True)
        item_count.columns = ['item_count']

        item_count_sort = item_count.sort_values(by = 'item_count', ascending=False)
        item_count_sort.reset_index(inplace=True)
        T = list(item_count_sort.item_id[-int(len(item_count_sort) * tail_ratio):].values)

        Tp = np.mean([sum([1 if item in T else 0 for item in self.pred_item[idx]]) / self.K
                            for idx in self.pred_item.index])
        return Tp

    def AveragePopularity(self) -> float:
        '''
        유저별로 주어진 추천 아이템 리스트의 평균 아이템 인기도 (self.popularity_df)
        -> 모든 유저들에 대한 평균 추천 아이템 인기도 (popularity_metric)
        '''
        popularity_metric = self.popularity_df.apply(lambda R: sum(R)).mean() / self.K

        return popularity_metric 
    
    def get_total_metrics(self) -> Dict:
        return {'Recall_K': self.Recall_K(),
                'MAP_K': self.MAP_K(),
                'Coverage': self.Coverage(),
                'TailPercentage': self.TailPercentage(),
                'AveragePopularity': self.AveragePopularity()
                }


class Qual_Metrics:
    '''
    Qualitative Metrics:
        - Diversity (Cosine, Jaccard)
        - Serendipity (PMI, Jaccard)
        - Novelty 
    
    * Rerank based on above metrics also available
    '''
    def __init__(self, dataset:Dataset, pred_item:Dict, pred_score:Dict, item_h_vector: Dict):
        self.pred_item = pd.Series(pred_item.values(), index=[int(k) for k in pred_item.keys()], 
                                    name='item_id').apply(lambda x: x.astype(int))
        self.pred_score = pd.Series(pred_score.values(), index=[int(k) for k in pred_score.keys()], 
                                    name='item_id') 
        self.n_user = dataset.n_user

        self.pmi_matrix = dataset.pmi_matrix
        self.jaccard_matrix = dataset.jaccard_matrix
        self.implicit_matrix = dataset.implicit_matrix 

        self.user_profiles = dataset.user_profiles #?? 
