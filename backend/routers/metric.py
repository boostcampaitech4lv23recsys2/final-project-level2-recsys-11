from typing import List, Dict
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations

from fastapi import APIRouter

from routers.data import dataset_info

router = APIRouter()

@router.get('/total_metrics')
async def total_configs_metrics():
    from routers.model import model_managers

    MODELS = model_managers.values()
    run = model_managers['EASE'].get_all_model_configs()[0]

    run_metrics = [("_".join(('EASE', run.string_key)), # 'EASE' = model.model_name
                    run.quantitative.Recall_K(),
                    run.quantitative.mapk(),
                    run.quantitative.NDCG(),
                    run.quantitative.AveragePopularity(),
                    run.quantitative.Coverage(),
                    run.quantitative.TailPercentage(),
                    run.qualitative.Total_Diversity().mean(),
                    run.qualitative.Total_Serendipity().mean(),
                    run.qualitative.Total_Novelty().mean())
                    # for model in MODELS for run in model.get_all_model_configs()
                    ] # model: ModelManager

    total_metrics_pd = pd.DataFrame(run_metrics,
                        columns=['Name', 'Recall', 'MAP', 'NDCG', 'AvgPopularity', 'Coverage', 'Tail_Percentage',
                                'Diversity', 'Serendipity', 'Novelty'
                                ])

    return total_metrics_pd.to_dict(orient='records')


@router.get('/rerank_metrics/{model_config_num}')
async def get_rerank_metrics(model_config_num: int, alpha: float, obj: str, mode: str, k: int=10):
    from routers.model import model_managers

    if model_config_num > 3:
        return ValueError('not a valid model config number')

    run = model_managers['EASE'].get_all_model_configs()[model_config_num]

    run_metrics = run.qualitative.total_rerank(alpha, obj, mode, k) # k

    return run_metrics


@router.get('/Diversity_for_users')
async def get_diversity_for_users():
    from routers.model import model_managers
    run = model_managers['EASE'].get_all_model_configs()[0]
    diversity_list = run.qualitative_model.Total_Diversity('jaccard')
    run_infos = [
        ("_".join(('EASE', run.string_key))),
        diversity_list,
        diversity_list.mean(),
        # TODO: 인풋으로 받은 color
    ]
    run_infos_pd = pd.DataFrame(
        run_infos, columns=['model', 'diversity', 'mean'] # , 'colors' 색 지정 하면 추가
    )
    
    return run_infos_pd.to_dict(orient='records')


@router.get('/qualitative/{model_config}')
async def get_qualitative_metrics():
    pass

@router.get('/quantitative')
async def get_quantitative_metrics(model_name:str, str_key:str):
    '''
    사용자가 원하는 실험에 대해 정량지표를 계산하고 이를 Return

    model_name(str): 모델 이름 (ex. BPR, EASE)
    str_key(str): 하이퍼 파라미터 값 (ex. negative_0.1_64_32)
    return metric_df(pd.DataFrame): columns=('model','recall','map','ndcg','avg_popularity','coverage)
    '''
    from routers.model import model_managers
    run = model_managers[model_name].get_model_config(str_key)

    run_metrics = [(
                run.quantitative.Recall_K(),
                run.quantitative.mapk(),
                run.quantitative.NDCG(),
                run.quantitative.AveragePopularity(),
                run.quantitative.Coverage())
                ]
    total_metrics_pd = pd.DataFrame(run_metrics,
                    columns=['recall','map','ndcg','avg_popularity','coverage'
                            ])
    return total_metrics_pd.to_dict(orient='records')


class quantitative_indicator:

    def __init__(self, dataset:dataset_info, pred_item:pd.Series): #, pred_score:pd.Series
        self.train_df = dataset.train_df
        self.ground_truth = dataset.ground_truth
        self.total_user = pred_item.index # 초기에 전체 유저
        self.K = dataset.K

        self.pred_item = pred_item.apply(lambda x: x[:self.K])
        # self.pred_score = pred_score.apply(lambda x: x[:self.K])

        self.n_user = dataset.n_user
        self.n_item = dataset.n_item

        # Popularity
        self.pop_user_per_item = dataset.pop_user_per_item # Dict
        self.pop_inter_per_item = dataset.pop_inter_per_item

        self.popularity_df = self.pred_item.apply(lambda R: [self.pop_user_per_item[int(item)] for item in R])

    # def change_pred_item(self, pred_item:pd.Series): # Reranking 후 pred_item이 바꾸고 싶을 때 사용하는 함수
    #     self.pred_item = pred_item

    def AveragePopularity(self) -> float:
        '''
        유저별로 주어진 추천 아이템 리스트의 평균 아이템 인기도 (self.popularity_df)
        -> 모든 유저들에 대한 평균 추천 아이템 인기도 (popularity_metric)
        '''
        popularity_metric = self.popularity_df.apply(lambda R: sum(R)).mean() / self.K

        return popularity_metric

    def apk(self, actual, predicted, k):
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

    def mapk(self,
        # TODO: users:np.array
        ):
        """
        Computes the mean average precision at k.
        This function computes the mean average prescision at k between two lists
        of lists of items.
        """
        return np.mean([self.apk(a, p, self.K) for a, p in zip(self.ground_truth.values, self.pred_item.values)])

    def NDCG(self, users=None) -> dict:
        '''
        NDCG = DCG / IDCG
        DCG = rel(i) / log2(i + 1)
        '''
        if users is None:
            users = self.total_user
        ndcg = {}
        for i in self.pred_item.loc[users].index: # TODO:self.pred_item.index -> self.pred_item.loc[self.selected_user].index
            k = min(self.K, len(self.ground_truth.loc[users,'item_id'].loc[i])) # TODO:len(self.ground_truth['item_id'].loc[1]) -> len(self.ground_truth.loc[self.selected_user,'item_id'].loc[i]) 
            idcg = sum([1 / np.log2(j + 2) for j in range(k)]) # 최대 dcg. +2는 range가 0에서 시작해서
            dcg = sum([int(self.pred_item[i][j] in set(self.ground_truth['item_id'].loc[i])) / np.log2(j + 2) for j in range(self.K)])
            ndcg[i] = dcg / idcg

        return ndcg

    def Coverage(self):
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

    def Recall_K(self, users=None) -> dict:
        if users is None:
            users = self.total_user

        actual = self.ground_truth.loc[users,'item_id']  # selected user만 Recall 계산
        predicted = self.pred_item.loc[users]
        recall = {}
        # true_users = 0

        for idx in actual.index:
            act_set = set(actual[idx].flatten())
            pred_set = set(predicted[idx][:self.K].flatten())
            if len(act_set) != 0:
                recall[idx] = len(act_set & pred_set) / min(len(act_set), len(pred_set))
                # true_users += 1

        return recall


class qualitative_indicator:

    def __init__(self, dataset:dataset_info, pred_item:pd.Series, pred_score:pd.Series):  #, item_h_matrix:np.array
        self.pred_item = pred_item
        self.pred_score = pred_score
        self.n_user = dataset.n_user

        # Serendipity
        self.pmi_matrix = dataset.pmi_matrix
        self.jaccard_matrix = dataset.jaccard_matrix
        self.implicit_matrix = dataset.implicit_matrix
        self.user_profiles = dataset.user_profiles
        # self.item_profiles = dataset.item_profiles

        # Diversity - jaccard
        self.genre = dataset.genre

        # Diversity - rating
        self.rating_matrix = dataset.rating_matrix
        self.item_mean_df = dataset.item_mean_df

        # Diversity - latent
        # self.item_h_matrix = item_h_matrix
        # self.item_item_matrix = self.item_h_matrix @ self.item_h_matrix.T

        # Diversity - dist dictionary
        self.rating_dist_dict = defaultdict(defaultdict)
        self.jaccard_dist_dict = defaultdict(defaultdict)
        self.latent_dist_dict = defaultdict(defaultdict)

        # Popularity
        self.pop_user_per_item = dataset.pop_user_per_item
        self.pop_inter_per_item = dataset.pop_inter_per_item

    def Total_Diversity(self, mode:str='jaccard', pred_item:pd.Series=None) -> np.array:
        '''
        모든 유저에 대한 추천 리스트를 받으면 각각의 Diversity 계산하여 리스트로 return
        mode : 사용할 방식. {'jaccard', 'rating', 'latent'}
        '''
        if pred_item is None:
            pred_item = self.pred_item
        DoA = np.array([0.5] + [self.Diversity(pred_item.loc[idx],mode) for idx in pred_item.index])  # 0번째는 패딩

        return DoA

    def Total_Serendipity(self, mode:str='PMI', pred_item:pd.Series=None) -> np.array:
        '''
        모든 유저에 대한 추천 리스트를 받으면 각각의 Serendipity를 계산하여 리스트로 return
        mode :  사용할 방식. {'PMI', 'jaccard'}
        '''
        if pred_item is None:
            pred_item = self.pred_item
        SoA = np.array([0.5] + [self.Serendipity(idx, pred_item.loc[idx], mode) for idx in pred_item.index])  # 0번째는 패딩

        return SoA

    def Total_Novelty(self, pred_item:pd.Series=None) -> np.array:
        '''
        모든 유저에 대한 추천 리스트를 받으면 각각의 Novelty를 계산하여 리스트로 return
        '''
        if pred_item is None:
            pred_item = self.pred_item
        NoA = np.array([0.5] + [self.Novelty(pred_item.loc[idx]) for idx in pred_item.index])  # 0번째는 패딩

        return NoA

    def Diversity(self, R:List[int], mode:str='jaccard'):
        '''
        R: 추천된 아이템 리스트
        mode: 사용할 방식. {'rating', 'jaccard', 'latent'}

        return: R의 diversity
        '''
        if mode == 'jaccard':
            dist_dict = self.jaccard_dist_dict
        elif mode == 'rating':
            dist_dict = self.rating_dist_dict
        elif mode == 'latent':
            dist_dict = self.latent_dist_dict
        # dist_dict = defaultdict(defaultdict)
        diversity = 0   # Diversity of User의 약자
        for i,j in combinations(R, 2):
            i,j = min(i,j), max(i,j)
            if i in dist_dict and j in dist_dict[i]:
                diversity += dist_dict[i][j]
            else:
                if mode == 'rating':             # mode 별로 하나씩 추가하면 될 듯
                    d = self.rating_dist(i,j)    # rating_dist 함수로 측정한 dist(i,j)
                elif mode == 'jaccard':
                    d = self.jaccard(i,j)
                elif mode == 'latent':
                    d = self.latent(i,j)
                dist_dict[i][j] = d
                diversity += d

        diversity /= ((len(R) * (len(R)-1)) / 2)

        return diversity

    def Serendipity(self, u:int, R:List[int], mode:str='PMI'):
        '''
        u: 유저 u
        R: u에게 추천된 아이템 리스트. 유저별로 그룹바이됨.
        mode: 사용할 방식. {'PMI', 'jaccard'}

        return: R의 serendipity
        '''
        user_pro = self.user_profiles[u]
        # user_pro = self.rating_matrix.T[self.rating_matrix.loc[u] != 0].index
        if mode == 'PMI':
            pmi_lst = self.pmi_matrix[R].loc[user_pro].min()
            return pmi_lst.mean()
        elif mode == 'jaccard':
            jac_lst = self.jaccard_matrix[R].loc[user_pro].min()
            return jac_lst.mean()
        else:
            raise ValueError("Only {PMI, jaccard} mode available")

    def Novelty(self, R:List[int]):
        lst = np.array([*map(lambda x: self.pop_user_per_item[x], R)])
        novelty = -np.log2(lst)
        return novelty.mean() / np.log2(self.n_user)

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

    # def latent(self, i:int, j:int):
    #     '''
    #     아이템 i,j latent vector들의 cosine similarity
    #     '''
    #     # if i == j:
    #     #     raise ValueError('i and j must be different items')

    #     norm_i = np.sqrt(np.square(self.item_h_matrix[i]).sum())
    #     norm_j = np.sqrt(np.square(self.item_h_matrix[j]).sum())
    #     similarity = self.item_item_matrix[i][j] / (norm_i * norm_j)

    #     return 1 - similarity



    def total_rerank(self, user_sample:list=None, alpha=0.5, obj='Serendipity', mode='PMI', k= 10):
        '''
        user_sample : reranking 할 user sample, default=None이면 전체 유저에 대해 실행
        alpha : [0,1] rel과 obj 간의 가중치
        obj : {'Diversity', 'Serendipity', 'Novelty'} 사용할 목적함수 
        mode : 사용할 모드. obj에 따라 값이 다름
        k : 만들 추천 리스트의 원소 갯수

        return : Diversity, Serendipity, Novelty가 담긴 딕트
        {
            'Diversity': Total_diversity한 값(ndarray),
            'Serendipity': Total_serendipity한 값(ndarray),
            'Novelty': Total_novelty한 값(ndarray)
        }
        '''
        total = pd.Series(dtype=object)
        if user_sample is None:
            for user in self.pred_item.index:
                total.loc[user] = self.rerank(user, alpha, obj, mode, k)
        else:
            for user in user_sample:
                total.loc[user] = self.rerank(user, alpha, obj, mode, k)
        ans = dict()
        ans['Diversity'] = self.Total_Diversity(pred_item=total) #어느 mode로 할지 정할 수 있게 하는 게 좋을 것 같기는 한데..
        ans['Serendipity'] = self.Total_Serendipity(pred_item=total)
        ans['Novelty'] = self.Total_Novelty(pred_item=total)
        return ans, total

    def rerank(self, user:int, alpha= 0.5, obj='Serendipity', mode='PMI', k= 10):
        '''
        user: 유저
        alpha : rel과 obj 간의 가중치
        obj : 사용할 목적함수 {'Diversity', 'Serendipity', 'Novelty'}
        mode : 사용할 모드. obj에 따라 값이 다름
        k : 만들 추천 리스트의 원소 갯수

        return : user에 대해 재정렬한 리스트 반환
        '''
        item = self.pred_item.loc[user]
        score = self.pred_score.loc[user]
        length = len(self.pred_item.loc[user]) # 이건 나중에 빼서 속도 개선하자. 처음부터 지정을 하던가 하는 식으로
        if k > length:
            raise ValueError('k should not be bigger than C')
        if obj == 'Diversity':
            if mode not in ['rating', 'jaccard', 'latent']:
                raise ValueError('only {rating, jaccard, latent} available for mode')
            if mode == 'rating':
                dist_function = self.rating_dist
            elif mode == 'jaccard':
                dist_function = self.jaccard
            elif mode == 'latent':
                dist_function = self.latent
            user_dict = defaultdict(float)
            C = list(item)
            for it, sc in zip(item,score):
                user_dict[it] = sc

            recommended_lst = []
            dist_dict = defaultdict(defaultdict)
            while len(recommended_lst) < k:
                if len(recommended_lst) == 0:
                    recommended_lst.append(C.pop(0))
                else:
                    best_score = 0
                    for i in C:
                        sum_of_dist = 0
                        for j in recommended_lst:
                            min_ij, max_ij = min(i,j), max(i,j)
                            if min_ij in dist_dict and max_ij in dist_dict[min_ij]:
                                sum_of_dist += dist_dict[min_ij][max_ij]
                            else:
                                dij = dist_function(min_ij,max_ij) # rating_dist, jaccard, latent
                                dist_dict[min_ij][max_ij] = dij
                                sum_of_dist += dij
                                # duplicate_cnt += 1
                        mean_of_dist = sum_of_dist / len(recommended_lst)
                        fobj = alpha * user_dict[i] + (1-alpha) * mean_of_dist
                        if fobj > best_score:
                            best_score, best_idx = fobj, i
                            
                    recommended_lst.append(best_idx)
                    C.remove(best_idx)
            
            return np.array(recommended_lst)

        elif obj == 'Serendipity':
            if mode == 'PMI':
                user_profile = self.rating_matrix.T[self.rating_matrix.loc[user] != 0].index
                obj_score = self.pmi_matrix[item].loc[user_profile].min()
                score = alpha * self.pred_score.loc[user] + (1 - alpha) * obj_score[item]
                topk_args = np.argsort(score.to_numpy())[:length-k-1:-1]
                recommended_lst = item[topk_args]
                return recommended_lst

            elif mode == 'jaccard':
                user_profile = self.rating_matrix.T[self.rating_matrix.loc[user] != 0].index
                obj_score = self.jaccard_matrix[item].loc[user_profile].min()
                score = alpha * self.pred_score.loc[user] + (1 - alpha) * obj_score[item]
                topk_args = np.argsort(score.to_numpy())[:length-k-1:-1]
                recommended_lst = item[topk_args]
                return recommended_lst
            else:
                raise ValueError('only {PMI, jaccard} available for mode')
        elif obj == 'Novelty':
            obj_score = np.log10(self.n_user / self.implicit_matrix[self.pred_item.loc[user]].sum())
            score = alpha * self.pred_score.loc[user] + (1 - alpha) * obj_score[item]
            topk_args = np.argsort(score.to_numpy())[:length-k-1:-1]
            recommended_lst = item[topk_args]
            return recommended_lst

        else:
            raise ValueError('only {Diversity, Serendipity, Novelty} available for objective function')