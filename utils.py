import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List
from typing import Dict
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations

from time import time

class dataset_info:
    def __init__(self, train_df, user, item_df, ground_truth, item_h_matrix): # 현재 이 클래스는 한 유저에 대해서 계산하는 클래스이나, 차라리 모든 유저를 받도록 하는 것이 나을 것.
        #user는 추천된 리스트를 받은 해당 유저

        train_df.columns = ['user_id', 'item_id', 'rating', 'timestamp', 'origin_timestamp']
        item_df.columns = ['item_id', 'movie_title', 'release_year', 'genre']

        self.train_df = train_df
        self.item_mean_df = train_df.groupby('item_id').agg('mean')['rating']
        self.rating_matrix = train_df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)

        # self.user_profile = {i: train_df[train_df['user_id:token'] == i]['item_id:token'].tolist() for i in train_df['user_id:token'].unique()}
        # 유저_id : 유저의 히스토리
        # ground_truth에 해당하는 정보는 빼야 할 수도?

        self.n_user = train_df['user_id'].nunique()
        self.user_profiles = {user: train_df[train_df['user_id'] == user]['item_id'].tolist() for user in train_df['user_id'].unique()} # 모든 유저들의 유저 프로파일로 수정
        self.item_profiles = {item : train_df[train_df['item_id'] == item]['user_id'].tolist() for item in train_df['item_id'].unique()}

        self.genre = dict()
        for i,j in zip(item_df['item_id'], item_df['genre']):
            self.genre[i] = j.split(' ')
        #결국 traindf는 받아야 하는 것 같기도. 그럼 상위 클래스를 만들기?

        # matrices for latent(i, j) 
        self.item_h_matrix = item_h_matrix
        self.item_item_matrix = self.item_h_matrix @ self.item_h_matrix.T

        # Popularity
        self.pop_of_each_items = self.calculate_Popularity()
        self.fam_of_each_items = self.calculate_Famousness()


    def calculate_Popularity(self):    # interaction 관점의 popularity
        '''

        지표를 계산하는 역할만 하는 함수인가?
        아니면 각 아이템의 인기도를 계산해 통째로 반환시키게 하는가?
        -> 일단 후자라고 생각하자 why? 어차피 train_df를 받아야 각 아이템이 상호작용된 횟수를 알 수 있다.

        return: 각 아이템 번호에 따른 인기도 딕트
        '''
        
        inter_count_of_items = self.train_df.groupby('item_id:token').count()['user_id:token']
        total_len = len(self.train_df)

        pop_of_each_items = dict()
        for i, j in zip(inter_count_of_items.keys(), inter_count_of_items):
            pop_of_each_items[i] = j / total_len
        #차라리 total_len을 항상 들고 다니고, 여기는 그냥 상호작용된 횟수만 넣고 다니는 게 좋은가?
        return pop_of_each_items


    def calculate_Famousness(self):   # 유저 관점의 popularity
        
        fam_of_each_items = (self.train_df['item_id:token'].value_counts() / self.n_user).to_dict()

        return fam_of_each_items
    

class quantitative_indicator():
    pass 


class qualitative_indicator:    
    def __init__(self, dataset_info:dataset_info, R_df:pd.DataFrame):  # dataset_info: class
        self.R_df = R_df
        self.n_user = dataset_info.n_user

        # Serendipity 
        self.user_profiles = dataset_info.user_profiles
        self.item_profiles = dataset_info.item_profiles
         
        # Diversity - jaccard
        self.genre = dataset_info.genre
        self.item_mean_df = dataset_info.item_mean_df

        # Diversity - rating
        self.rating_matrix = dataset_info.rating_matrix

        # Diversity - latent
        self.item_h_matrix = dataset_info.item_h_matrix
        self.item_item_matrix = dataset_info.item_item_matrix

        # Popularity
        self.pop_of_each_items = dataset_info.pop_of_each_items
        self.fam_of_each_items = dataset_info.fam_of_each_items


    def Diversity(self, R:List[int], mode:str='jaccard'):
        '''
        R: 추천된 아이템 리스트
        mode: 사용할 방식. {'rating', 'jaccard', 'latent'}

        return: R의 diversity
        '''
        if mode == 'rating':
            dist_dict = defaultdict(defaultdict)
            DoU = 0   # Diversity of User의 약자
            for i,j in combinations(R, 2):
                i,j = min(i,j), max(i,j)
                if i in dist_dict and j in dist_dict[i]:
                    DoU += dist_dict[i][j]
                else:
                    if mode == 'rating':             # mode 별로 하나씩 추가하면 될 듯
                        d = self.rating_dist(i,j)    # rating_dist 함수로 측정한 dist(i,j)
                    dist_dict[i][j] = d
                    DoU += d
            DoU /= ((len(R) * (len(R)-1)) / 2)

            return DoU

        else:
            diversity = 0
            for i in R:
                for j in R:
                    if i == j: continue
                    dist = eval('self.'+ mode)(i,j)
                    diversity += dist
            diversity = diversity / (len(R) * (len(R) - 1))
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
        # p(i) = popularity
        # 1 - p(i) or -log(p(i))

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


    def Serendipity_total(self, i:int, mode:str='PMI'): # 모든 유저들에 대한 Serendipity_foreach(i, mode)
        pass


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
        
        norm_i = np.sqrt(np.square(self.B[i]).sum())
        norm_j = np.sqrt(np.square(self.B[j]).sum())
        similarity = self.item_similarity[i][j] / (norm_i * norm_j)

        return 1 - similarity



#유저 1에 대해서 실험해보자

train_df = pd.read_csv('/opt/ml/final-project-level2-recsys-11/dataset/ml-1m/ml-1m.inter', sep='\t')
item_df = pd.read_csv('/opt/ml/final-project-level2-recsys-11/dataset/ml-1m/ml-1m.item', sep='\t')
rec_df = pd.read_csv('/opt/ml/input/final-project-level2-recsys-11/RecBole/inference/EASE_9e463a68-2033-47dc-bac6-d6ee82df8e91.csv', sep='\t')

tmp = qualitative_indicator(train_df, 1, item_df)
R = rec_df[:10]['item'].tolist()

start = time()

print('유저 1에 대한 추천리스트 R', R)
print('R의 Serendipity by PMI', tmp.Serendipity(R, 'PMI'))
print('R의 Serendipity by jaccard', tmp.Serendipity(R, 'jaccard'))
print('R의 Novelty', tmp.Novelty(R))

end = time()

print('시간 소모', end - start)