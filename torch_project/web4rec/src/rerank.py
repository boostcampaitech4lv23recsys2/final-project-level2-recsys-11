from tqdm import tqdm
from copy import deepcopy

import pandas as pd
import numpy as np


def get_total_reranks(
    mode: str,
    candidates,
    prediction_mat,
    distance_mat,
    user_profile,
    item_popularity,
    alpha=0.5,
    k=10
):

    if 'diversity' in mode:
        return Rerank.diverity(
            candidates=candidates,
            prediction_mat=prediction_mat,
            distance_mat=distance_mat,
            alpha=alpha,
            k=k
        )
    elif 'serendipity' in mode:
        return Rerank.serendipity(
            candidates=candidates,
            prediction_mat=prediction_mat,
            distance_mat=distance_mat,
            user_profile=user_profile,
            alpha=alpha,
            k=k
        )
    elif 'novelty' in mode:
        return Rerank.novelty(
            candidates=candidates,
            prediction_mat=prediction_mat,
            item_popularity=item_popularity,
            alpha=alpha,
            k=k
        )


class Rerank:
    def diverity(candidates, prediction_mat, distance_mat, alpha, k):
        reranks = []
        for uidx, candidate in tqdm(enumerate(candidates), total=len(candidates), desc='[W4R] '):
            rerank = [candidate[0]]
            temp_candidate = deepcopy(candidate[1:])
            while len(rerank) < 10:
                obj_scores = distance_mat[temp_candidate][:, rerank].sum(axis=1)
                scores = alpha * prediction_mat[uidx, temp_candidate] + (1 - alpha) * (obj_scores / len(rerank))
                temp_candidate = temp_candidate[np.argsort(-scores)]
                rerank.append(temp_candidate[0])
                temp_candidate = temp_candidate[1:]
            reranks.append(rerank)
        return np.array(reranks)


    def serendipity(candidates, prediction_mat, user_profile, distance_mat, alpha, k):
        reranks = []
        for uidx, (candidate, profile) in tqdm(enumerate(zip(candidates, user_profile)), total=len(candidates), desc='[W4R] '):
            obj_scores = distance_mat[candidate, :][:, profile].min(axis=1)
            scores = alpha * prediction_mat[uidx, candidate] + (1 - alpha) * obj_scores
            reranks.append(candidate[np.argsort(-scores)][:k])
        return np.array(reranks)


    def novelty(candidates, prediction_mat, item_popularity, alpha, k):
        reranks = []
        for uidx, candidate in tqdm(enumerate(candidates), total=len(candidates), desc='[W4R] '):
            obj_scores = -np.log10(item_popularity[candidate])
            scores = alpha * prediction_mat[uidx, candidate] + (1 - alpha) * obj_scores
            reranks.append(candidate[np.argsort(-scores)][:k])
        return np.array(reranks)
