import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from typing import Dict


class quantitative_indicator():
    pass

class qualitative_indicator():
    def __init__(self, train_df, user, item_df, item_h_matrix):
        self.train_df = train_df

        # self.user_profile = {i: train_df[train_df['user_id:token'] == i]['item_id:token'].tolist() for i in train_df['user_id:token'].unique()}
        # 유저_id : 유저의 히스토리
        # ground_truth에 해당하는 정보는 빼야 할 수도?

        self.user_profile = train_df[train_df['user_id:token'] == user]['item_id:token'].tolist()
        #user는 추천된 리스트를 받은 해당 유저

        self.genre = dict()
        for i,j in zip(item_df['item_id:token'], item_df['genre:token_seq']):
            self.genre[i] = j.split(' ')

        #결국 traindf는 받아야 하는 것 같기도. 그럼 상위 클래스를 만들기?

        # matrices for latent(i, j) 
        self.item_h_matrix = item_h_matrix
        self.item_item_matrix = self.item_h_matrix @ self.item_h_matrix.T


    def Diversity(self, R:List[int], mode:str='jaccard'):
        '''
        R: 추천된 아이템 리스트
        mode: 사용할 방식. {'rating', 'jaccard', 'latent'}

        return: R의 diversity
        '''
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
        return 


    def Popularity(): 


    def Novelty():
        pass


    def Serendipity_foreach(self, i:int, mode:str='PMI'):
        '''
        i: 아이템 i
        mode: 사용할 방식. {'PMI', 'jaccard'}

        return: 유저 프로필과 i 간의 PMI 최소값

        이 함수는 추후 reranking에 그대로 쓰입니다. 각 i의 serendipity 값을 계산해주죠!
        '''
        min_seren = np.inf
        for item in self.user_profile:
            seren = eval('self.'+ mode)(i, item)
            min_seren = min(min_seren, seren)
        return min_seren


    def PMI(self, i:int, j:int):
        '''
        i: 아이템 i
        j: 아이템 j

        return: i와 j의 scaled PMI 값(0~1)
        '''

        total_user = self.train_df['user_id:token'].nunique()
        set_i = set(self.train_df[self.train_df['item_id:token'] == i]['user_id:token'])
        set_j = set(self.train_df[self.train_df['item_id:token'] == j]['user_id:token'])
        p_i = len(set_i) / total_user
        p_j = len(set_j) / total_user
        p_ij = len(set_i & set_j) / total_user

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

print('유저 1에 대한 추천리스트 R', R)
print('R의 Serendipity by PMI', tmp.Serendipity(R, 'PMI'))
print('R의 Serendipity by jaccard', tmp.Serendipity(R, 'jaccard'))