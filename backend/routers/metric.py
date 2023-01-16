from typing import List
import pandas as pd
import numpy as np 
from collections import defaultdict
from itertools import combinations

from fastapi import APIRouter

from routers.data import dataset_info
from routers import model

router = APIRouter()

@router.get('testing2')
def testing2():
    return model.testing.datetime

class quantitative_indicator():

    def __init__(self, dataset_info:dataset_info, pred_item:pd.Series, pred_score:pd.DataFrame):
        self.train_df = dataset_info.train_df
        self.ground_truth = dataset_info.ground_truth

        self.K = dataset_info.K

        self.pred_item = pred_item # 전체 추천 리스트들. 유저가 인덱스이고 한 컬럼에 모든 각 유저에 대한 추천리스트가 담김
        self.pred_score = pred_score 
        
        self.n_user = dataset_info.n_user
        self.n_item = dataset_info.n_item

        # Popularity
        self.pop_user_per_item = dataset_info.pop_user_per_item
        self.pop_inter_per_item = dataset_info.pop_inter_per_item
        self.popularity_df = self.pred_item.apply(lambda R: [self.pop_user_per_item[item] for item in R])
        
    def AveragePopularity(self) -> float:
        '''
        유저별로 주어진 추천 아이템 리스트의 평균 아이템 인기도 (self.popularity_df)
        -> 모든 유저들에 대한 평균 추천 아이템 인기도 (popularity_metric)
        '''
        popularity_metric = self.popularity_df.apply(lambda R: sum(R)).mean() / self.K

        return popularity_metric 

    def apk(self, actual, predicted, k=10):
        """
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

    def mapk(self, actual, predicted, k=10):
        """
        Computes the mean average precision at k.
        This function computes the mean average prescision at k between two lists
        of lists of items.
        """
        return np.mean([self.apk(a, p, k) for a, p in zip(actual, predicted)])

    def NDCG(self):
        '''
        NDCG = DCG / IDCG
        DCG = rel(i) / log2(i + 1)
        '''
        ndcg = 0
        for i in self.pred_item.index:
            k = min(self.K, len(self.ground_truth['item'].loc[1]))
            idcg = sum([1 / np.log2(j + 2) for j in range(k)]) # 최대 dcg. +2는 range가 0에서 시작해서
            dcg = sum([int(self.pred_item.loc[i].item()[j] in set(self.ground_truth['item'].loc[i])) / np.log2(j + 2) for j in range(self.K)])
            ndcg += dcg / idcg
        return ndcg / len(self.pred_item)

    def Coverage(self):
        '''
        return: 추천된 아이템의 고유값 수 / 전체 아이템 수
        '''
        rec_num = self.rec_df['item'].nunique()
        #이 TOTAL은 GROUND까지 포함한 값이어야 한다.
        return rec_num / self.n_item

    def Recall(self):
        #해당 코드는 현재 hit지표에 맞게 쓰여저 있습니다.
        sum_recall = 0
        pass

    def TailPercentage(self, tail_ratio=0.1):
        item_count = self.train_df.groupby('item_id').agg('count')
        item_count.drop(['rating', 'timestamp','origin_timestamp'], axis=1, inplace=True)
        item_count.columns =  ['item_count']

        item_count_sort = item_count.sort_values(by = 'item_count', ascending=False)
        item_count_sort.reset_index(inplace=True)
        T = item_count_sort.item_id[-int(len(item_count_sort) * tail_ratio):].values

        Tp = np.mean([sum([1 if item in T else 0 for item in self.pred_item.loc[idx,'item']]) / self.K for idx in self.pred_item.index])
        return Tp

    def Recall_K(self):
        # def recall_at_k(actual, predicted, topk):
        topk = self.K
        pred_item = self.pred_item                              # 유저, [추천리스트] 형태
        ground_truth = self.ground_truth
        T_df = ground_truth.groupby('user').agg(list) # pred_item와 같은 형태
        actual = T_df.item
        predicted = pred_item.item
        sum_recall = 0.0
        true_users = 0
        for idx in actual.index:
            act_set = set(actual[idx])
            pred_set = set(predicted[idx][:topk])
            if len(act_set) != 0:
                sum_recall += len(act_set & pred_set) / float(len(act_set))
                true_users += 1

        return sum_recall / true_users


class qualitative_indicator:

    def __init__(self, dataset_info:dataset_info, pred_item:pd.DataFrame):  # dataset_info: class
        self.pred_item = pred_item
        self.n_user = dataset_info.n_user

        # Serendipity
        self.user_profiles = dataset_info.user_profiles
        self.item_profiles = dataset_info.item_profiles

        # Diversity - jaccard
        self.genre = dataset_info.genre
        
        # Diversity - rating
        self.rating_matrix = dataset_info.rating_matrix
        self.item_mean_df = dataset_info.item_mean_df

        # Diversity - latent
        self.item_h_matrix = dataset_info.item_h_matrix
        self.item_item_matrix = dataset_info.item_item_matrix

        # Diversity - dist dictionary
        self.dist_dict = defaultdict(defaultdict)

        # Popularity
        self.pop_of_each_items = dataset_info.pop_of_each_items
        self.fam_of_each_items = dataset_info.fam_of_each_items

    def Total_Diversity(self, mode:str='jaccard') -> List[float]:
        '''
        모든 유저에 대한 추천 리스트를 받으면 각각의 Diversity 계산하여 리스트로 return
        mode : 사용할 방식. {'jaccard', 'rating', 'latent'}
        '''
        DoA = [0.5] + [self.Diversity(self.pred_item.loc[idx, 'item'],mode) for idx in self.pred_item.index]  # 0번째는 패딩

        return DoA

    def Total_Serendipity(self, mode:str='jaccard') -> List[float]:
        '''
        모든 유저에 대한 추천 리스트를 받으면 각각의 Serendipity를 계산하여 리스트로 return
        mode :  사용할 방식. {'PMI', 'jaccard'}
        '''
        SoA = [0.5] + [self.Serendipity(self.pred_item.loc[idx, 'item'],mode) for idx in self.pred_item.index]  # 0번째는 패딩

        return SoA

    def Total_Novelty(self) -> List[float]:
        '''
        모든 유저에 대한 추천 리스트를 받으면 각각의 Novelty를 계산하여 리스트로 return
        '''
        NoA = [0.5] + [self.Novelty(self.pred_item.loc[idx, 'item']) for idx in self.pred_item.index]  # 0번째는 패딩

        return NoA

    def Diversity(self, R:List[int], mode:str='jaccard'):
        '''
        R: 추천된 아이템 리스트
        mode: 사용할 방식. {'rating', 'jaccard', 'latent'}

        return: R의 diversity
        '''
        # dist_dict = defaultdict(defaultdict)
        diversity = 0   # Diversity of User의 약자
        for i,j in combinations(R, 2):
            i,j = min(i,j), max(i,j)
            if i in self.dist_dict and j in self.dist_dict[i]:
                diversity += self.dist_dict[i][j]
            else:
                if mode == 'rating':             # mode 별로 하나씩 추가하면 될 듯
                    d = self.rating_dist(i,j)    # rating_dist 함수로 측정한 dist(i,j)
                elif mode == 'jaccard':
                    d = self.jaccard(i,j)
                elif mode == 'latent':
                    d = self.latent(i,j)
                self.dist_dict[i][j] = d
                diversity += d
            diversity /= ((len(R) * (len(R)-1)) / 2)

            return diversity

    def Serendipity(self, R:List[int], mode:str='PMI'):
        '''
        R: 추천된 아이템 리스트
        mode: 사용할 방식. {'PMI', 'jaccard'}

        return: R의 serendipity
        '''

        sum_pmi = 0
        for i in R:
            sum_pmi += self.Serendipity_foreach(i, mode)
        serendipity = sum_pmi / len(R)
        return serendipity

    def Novelty(self, R:List[int]):
        lst = np.array([*map(lambda x: self.fam_of_each_items[x], R)])
        novelty = -np.log2(lst)
        return novelty.mean() / np.log2(self.total_user)

    def Serendipity_foreach(self, i:int, u:int, mode:str='PMI'):
        '''
        i: 아이템 i
        u: 유저 u
        mode: 사용할 방식. {'PMI', 'jaccard'}

        return: 유저 프로필과 i 간의 PMI 최소값

        이 함수는 추후 reranking에 그대로 쓰입니다. 각 i의 serendipity 값을 계산해주죠!
        '''
        min_seren = np.inf
        for item in self.user_profile[u]:
            seren = eval('self.'+ mode)(i, item)
            min_seren = min(min_seren, seren)
        return min_seren

    def PMI(self, i:int, j:int):
        '''
        i: 아이템 i
        j: 아이템 j

        return: i와 j의 scaled PMI 값(0~1)
        '''
        set_i = set(self.item_profiles[i])
        set_j = set(self.item_profiles[j])
        p_i = len(set_i) / self.n_user
        p_j = len(set_j) / self.n_user
        p_ij = len(set_i & set_j) / self.n_user

        #pmi 공식. -1~1 사이의 값
        pmi = np.log2(p_ij / (p_i * p_j)) / -np.log2(p_ij)
        return (1 - pmi) / 2

    def jaccard(self, i:int, j:int):
        '''
        i: 아이템 i
        j: 아이템 j

        return: i와 j의 jaccard 값(0~1)
        '''
        s1 = set(self.genre[i])
        s2 = set(self.genre[j])

        return 1 - len(s1 & s2) / len(s1 | s2)

    def rating_dist(self, i:int, j:int):
        '''
        i: 아이템 i
        j: 아이템 j

        return : i와 j의 rating 기반 유사도 값
        '''
        A = self.rating_matrix
        item_mean_df = self.item_mean_df

        a = A.loc[(A.loc[:,i] * A.loc[:,j]) != 0, i] - item_mean_df[i] # rui - mean(ri)
        b = A.loc[(A.loc[:,i] * A.loc[:,j]) != 0, j] - item_mean_df[j] # ruj - mean(rj)
        sum_of_A = sum(a*b)  # 분자 식
        sum_of_B = sum(a**2) # 분모 식 앞 부분
        sum_of_C = sum(b**2) # 분모 식 뒷 부분

        result = 0.5 -  (sum_of_A / (2 * np.sqrt(sum_of_B) * np.sqrt(sum_of_C)))

        return result

    def latent(self, i:int, j:int):
        '''
        아이템 i,j latent vector들의 cosine similarity
        '''
        # if i == j:
        #     raise ValueError('i and j must be different items')

        norm_i = np.sqrt(np.square(self.item_h_matrix[i]).sum())
        norm_j = np.sqrt(np.square(self.item_h_matrix[j]).sum())
        similarity = self.item_item_matrix[i][j] / (norm_i * norm_j)

        return 1 - similarity

