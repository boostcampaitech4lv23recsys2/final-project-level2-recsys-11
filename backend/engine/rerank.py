from tqdm import tqdm
from copy import deepcopy

import pandas as pd
import numpy as np


async def get_total_reranks(
    mode: str,
    candidates,
    prediction_mat,
    distance_mat,
    user_profile,
    item_popularity,
    alpha,
    k=10
):

    if 'diversity' in mode:
        return Rerank.diversity(
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
    def diversity(candidates: pd.Series, prediction_mat, distance_mat, alpha, k):
        reranks = {}
        for uidx, candidate in tqdm(candidates.items(), total=len(candidates), desc='[W4R] '):
            rerank = [candidate[0]]
            temp_candidate = np.array(candidate[1:])
            while len(rerank) < k:
                obj_scores = distance_mat[temp_candidate][:, rerank].sum(axis=1)
                scores = alpha * prediction_mat[uidx, temp_candidate] + (1 - alpha) * (obj_scores / len(rerank))
                # print(prediction_mat[uidx, temp_candidate])
                temp_candidate = temp_candidate[np.argsort(-scores)]
                rerank.append(temp_candidate[0])
                temp_candidate = temp_candidate[1:]
            reranks[uidx] = rerank
        return pd.Series(reranks)


    def serendipity(candidates: pd.Series, user_profile: pd.Series, prediction_mat, distance_mat, alpha, k, n_profiles=100):
        reranks = {}
        for uidx, candidate in tqdm(candidates.items(), total=len(candidates), desc='[W4R] '):
            obj_scores = distance_mat[candidate, :][:, user_profile[uidx][:n_profiles]].min(axis=1)
            scores = alpha * prediction_mat[uidx, candidate] + (1 - alpha) * obj_scores
            candidate = np.array(candidate)
            reranks[uidx] = candidate[np.argsort(-scores)][:k]
        return pd.Series(reranks)


    def novelty(candidates: pd.Series, item_popularity, prediction_mat, alpha, k):
        reranks = {}
        for uidx, candidate in tqdm(candidates.items(), total=len(candidates), desc='[W4R] '):
            obj_scores = -np.log10(item_popularity[candidate])
            scores = alpha * prediction_mat[uidx, candidate] + (1 - alpha) * obj_scores
            candidate = np.array(candidate)
            reranks[uidx] = candidate[np.argsort(-scores)][:k]
        return pd.Series(reranks)
