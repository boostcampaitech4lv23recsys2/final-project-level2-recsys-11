import random
import math
import os
from collections import deque

import torch
import numpy as np
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class NegativeSampler:
    def __init__(self, pos_pair:pd.DataFrame, n_negs:int=2, mode: str='uni'):
        self.n_negs = n_negs
        self.mode = mode

        user_id = pos_pair.columns[0]
        item_id = pos_pair.columns[1]

        self.user_positives = pos_pair.groupby(user_id)[item_id].apply(list).sort_index()
        
        self.total_items = pos_pair[item_id].unique()
        self.user_negatives = {}
        for u, pos_list in self.user_positives.iteritems():
            neg_list = list(set(self.total_items) - set(pos_list))
            self.user_negatives[u] = neg_list

        self.user_negatives = pd.Series(self.user_negatives).sort_index()

        # user_negatives 에서 개수를 확 늘리는 방법이 있다.
        if mode == 'pop':
            self.item_popularity = pos_pair[item_id].value_counts() / len(pos_pair)
            self.item_popularity **= 0.4
            self.item_popularity //= self.item_popularity.min()

            for u, neg_list in self.user_negatives.iteritems():
                pop_neg_list = np.repeat(neg_list, self.item_popularity.loc[neg_list])
                self.user_negatives.loc[u] = pop_neg_list


    def sampling(self) -> pd.DataFrame:
        neg_samples = {}
        if self.mode == 'uni':
            for u, neg_list in self.user_negatives.iteritems():
                neg_samples[u] = deque()
                for _ in range(len(self.user_positives.loc[u])):
                    neg_samples[u].extend(random.sample(neg_list, k=self.n_negs))

        elif self.mode == 'pop':
            for u, neg_list in self.user_negatives.iteritems():
                neg_samples[u] = deque()
                for _ in range(len(self.user_positives.loc[u])):
                    neg_sample = []
                    while len(neg_sample) != self.n_negs:
                        neg_item = neg_list[np.random.randint(len(neg_list))]
                        if neg_item not in neg_sample:
                            neg_sample.append(neg_item)
                    neg_samples[u].extend(neg_sample)
        
        return neg_samples


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
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


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg
    return res / float(len(actual))

def get_full_sort_score(answers, pred_list): # baseline trainer에 있는 것 그대로 가져옴
    recall, ndcg = [], []
    for k in [5, 10]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))
    post_fix = {
        "RECALL@5": "{:.4f}".format(recall[0]),
        "NDCG@5": "{:.4f}".format(ndcg[0]),
        "RECALL@10": "{:.4f}".format(recall[1]),
        "NDCG@10": "{:.4f}".format(ndcg[1]),
    }
    print(post_fix, flush=True)

    return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)