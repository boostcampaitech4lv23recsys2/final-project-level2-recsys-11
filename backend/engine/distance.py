import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, linalg


async def get_distance_mat(train_interaction: pd.DataFrame):

    # Diversity cos distance
    train_mat = csr_matrix((np.ones(len(train_interaction)), (train_interaction['user_idx'], train_interaction['item_idx'])))
    train_mat = train_mat / linalg.norm(train_mat, axis=1).reshape(-1, 1)
    # (1)
    item_sym_mat = (train_mat.T @ train_mat)

    # (2)
    item_norm_arr = np.linalg.norm(train_mat, axis=0)
    item_norm_mat = item_norm_arr.reshape(-1, 1) @ item_norm_arr.reshape(1, -1)

    # (1) / (2)
    item_sim_mat = np.asarray(item_sym_mat / (item_norm_mat + 1e-8))

    # 1/2 등을 안하는 이유는 어차피 양수 벡터의 코사인 시밀러리티는 항상 양수임
    cos_matrix = 1 - item_sim_mat

    # Serendipity PMI distance
    # p(i)
    p_each_arr = np.asarray(train_mat.sum(axis=0)) / train_interaction['user_idx'].nunique()

    # p(i, j)
    p_pair_mat = (train_mat.T @ train_mat) / train_interaction['user_idx'].nunique()

    # PMI log2(p(i, j) / (p(i) * p(j)))  /  log2(p(i, j))

    # p(i) * p(j)
    p_each_mat = p_each_arr.reshape(-1, 1) @ p_each_arr.reshape(1, -1)

    PMI_nominator_mat = np.log2(np.asarray(p_pair_mat / (p_each_mat + 1e-9)) + 1e-9)
    PMI_mat = PMI_nominator_mat / - (np.log2(np.asarray(p_pair_mat) + 1e-9) + 1e-9)

    pmi_matrix = (1 - PMI_mat) / 2
    return cos_matrix, pmi_matrix


async def get_jaccard_mat(jac_vector: pd.Series):
    jac = np.array(jac_vector.sort_index().tolist())

    nominator = jac @ jac.T #분자가 되는 부분
    denominator = np.triu(nominator) # 분모가 되는 부분
    denominator = denominator - denominator.diagonal()
    denominator = -(denominator + denominator.T) # -(P(A&B) - P(A) - P(B))
    jac_mat = 1 - (nominator / (denominator + 1e-9)) # 자카드 행렬
    np.fill_diagonal(jac_mat, val=0)
    return jac_mat