import numpy as np
import pandas as pd

class Quantative:

    def recall_at_k(predicted: list, actual: list, k=10):
        predicted = predicted[:k]
        denominator = len(set(actual) & set(predicted)) 
        nominator = min(len(predicted), len(actual)) 
        return denominator / nominator


    def average_precision_at_k(predicted: list, actual: list, k=10):
        predicted = predicted[:k]
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[: i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        if not actual:
            return 0.0
        return score / min(len(actual), k)


    def ndcg_at_k(predicted: list, actual: list, k=10):
        k2 = min(k, len(actual))
        idcg = sum([1. / np.log2(i + 2) for i in range(k2)]) # 최대 dcg. +2는 range가 0에서 시작해서
        dcg = sum([int(predicted[i] in set(actual)) / np.log2(i + 2) for i in range(k)])
        return dcg / idcg


    def tail_percentage(predicted: list, tail_items: list, k=10):
        predicted = predicted[:k]
        return len(set(predicted) & set(tail_items)) / len(set(predicted))

    
    def average_popularity(predicted: list, item_popularity: pd.Series, k=10): 
        predicted = predicted[:k]
        return item_popularity[predicted].mean()

    
    def coverage(total_preds: list, total_items: list): #0.35
        return len(set(total_preds)) / len(set(total_items))


class Qualitative:

    def diversity(predicted: list, distance_mat: np.ndarray, k=10):
        predicted = predicted[:k]
        return distance_mat[predicted, :][:, predicted].sum() / (len(predicted) * (len(predicted) - 1))
        

    def serendipity(predicted: list, profiles: list, distance_mat: np.ndarray, k=10):
        predicted = predicted[:k]
        ret = 0
        for pred in predicted:
            ret += distance_mat[pred, profiles].min()
        return ret / len(predicted)


    def novelty(predicted: list, n_users: int, item_popularity: pd.Series, k=10):
        predicted = predicted[:k]

        ret = 0
        for pred in predicted:
            ret -= np.log2(item_popularity[pred])
        nov_max = np.log2(n_users)
        return ret / (nov_max * len(predicted))

# metirces
def get_total_information(
    predicts,
    actuals,
    cos_dist,
    pmi_dist,
    user_profile,
    item_popularity,
    tail_items,
    total_items,
    jac_dist=None,
    k=10
):
    metrices = {
        'recall': [0 for _ in range(predicts.shape[0])],
        'ndcg': [0 for _ in range(predicts.shape[0])],
        'avg_precision': [0 for _ in range(predicts.shape[0])],
        'avg_popularity': [0 for _ in range(predicts.shape[0])],
        'tail_percentage': [0 for _ in range(predicts.shape[0])],
        'diversity_cos': [0 for _ in range(predicts.shape[0])],
        'serendipity_pmi': [0 for _ in range(predicts.shape[0])],
        'novelty': [0 for _ in range(predicts.shape[0])]
    }


    # recall, ndcg, apk
    for i, (predicted, actual) in enumerate(zip(predicts, actuals)):
        metrices['recall'][i] = Quantative.recall_at_k(predicted, actual, k=k)

    for i, (predicted, actual) in enumerate(zip(predicts, actuals)):
        metrices['ndcg'][i] = Quantative.ndcg_at_k(predicted, actual, k=k)
        
    for i, (predicted, actual) in enumerate(zip(predicts, actuals)):
        metrices['avg_precision'][i] = Quantative.average_precision_at_k(predicted, actual, k=k)


    # avg_pop, tail_percentage
    for i, predicted in enumerate(predicts):
        metrices['avg_popularity'][i] = Quantative.average_popularity(predicted, item_popularity, k=k)

    for i, predicted in enumerate(predicts):
        metrices['tail_percentage'][i] = Quantative.tail_percentage(predicted, tail_items, k=k)


    # diversity, serendipity, novelty
    for i, predicted in enumerate(predicts):
        metrices['diversity_cos'][i] = Qualitative.diversity(predicted, cos_dist, k=k)

    for i, predicted in enumerate(predicts):
        metrices['serendipity_pmi'][i] = Qualitative.serendipity(predicted, user_profile[i], pmi_dist, k=k)

    for i, predicted in enumerate(predicts):
        metrices['novelty'][i] = Qualitative.novelty(predicted, len(user_profile), item_popularity, k=k)

    if jac_dist is not None:
        metrices['diversity_jac'] = [0 for _ in range(predicts.shape[0])]
        metrices['serendipity_jac'] = [0 for _ in range(predicts.shape[0])]

        for i, predicted in enumerate(predicts):
            metrices['diversity_jac'][i] = Qualitative.diversity(predicted, jac_dist, k=k)
        for i, predicted in enumerate(predicts):
            metrices['serendipity_jac'][i] = Qualitative.serendipity(predicted, user_profile[i], jac_dist, k=k)

    metric_per_user = pd.Series(metrices)
    metrices = metric_per_user.apply(np.mean)
    metrices['coverage'] = Quantative.coverage(predicts[:, :k].flatten(), total_items)

    return metrices, metric_per_user