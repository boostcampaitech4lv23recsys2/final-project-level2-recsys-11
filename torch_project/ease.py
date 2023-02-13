from tqdm import tqdm
import random
import pickle
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from utils import NegativeSampler, set_seed, get_full_sort_score

from web4rec.web4rec import Web4Rec, Web4RecDataset

from scipy.sparse import csr_matrix, linalg


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data/ml-1m')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_valids', default=1, type=int)
    
    
    parser.add_argument("--exp_name", type=str)

    parser.add_argument("--weight_decay", default=200, type=float)

    args = parser.parse_args()
    return args


def main(args):
    ratings = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))
    ratings = ratings.sort_values(by=['user_id', 'timestamp'], ignore_index=True)

    n_users = ratings['user_id'].nunique()
    n_items = ratings['item_id'].nunique()
    
    user2idx = {v: k for k, v in enumerate(ratings['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(ratings['item_id'].unique())}

    idx2user = {k: v for k, v in enumerate(ratings['user_id'].unique())}
    idx2item = {k: v for k, v in enumerate(ratings['item_id'].unique())}

    # Web4Rec - Dataset Process start
    Web4Rec.login(token='mkdir_token')

    ground_truth = pd.read_csv(os.path.join(args.data_dir, 'test_ratings.csv'))
    user_side = pd.read_csv(os.path.join(args.data_dir, 'users.csv'))
    item_side = pd.read_csv(os.path.join(args.data_dir, 'items.csv'))

    item_side['genres'] = item_side['genres'].apply(lambda s: s.replace('|', ' '))
    item_side.rename(
        columns= {
            'title': 'item_name',
            'genres': 'genres:multi',
            }, 
        inplace=True
    )

    w4r_dataset = Web4RecDataset(dataset_name='ml-1m')
    w4r_dataset.add_train_interaction(ratings)
    w4r_dataset.add_ground_truth(ground_truth)
    w4r_dataset.add_user_side(user_side)
    w4r_dataset.add_item_side(item_side)

    Web4Rec.register_dataset(w4r_dataset)
    # Web4Rec - Dataset Process end


    ratings['user_idx'] = ratings['user_id'].map(user2idx)
    ratings['item_idx'] = ratings['item_id'].map(item2idx)

    # leave n out -> valid answers
    valid_ratings = ratings.groupby('user_id').tail(args.n_valids)
    valid_answers = valid_ratings.groupby('user_id')['item_id'].apply(list).values.tolist()
    train_ratings = ratings.drop(valid_ratings.index)
    

    train_data = train_ratings[['user_idx', 'item_idx']]

    
    X = csr_matrix((np.ones(len(train_data)), (train_data['user_idx'], train_data['item_idx'])))
    G = X.T @ X
    G[np.diag_indices(G.shape[0])] += args.weight_decay
    P = np.linalg.inv(G.todense())

    B = np.asarray(P / -np.diag(P))
    B[np.diag_indices(B.shape[0])] = 0

    total_pred_rating_mat = X @ B

    total_pred_rating_mat2 = total_pred_rating_mat.copy()
    total_pred_rating_mat2[train_data['user_idx'].values, train_data['item_idx'].values] = 0.0

    preds = np.argsort(-total_pred_rating_mat2, axis=1)

    preds = np.vectorize(lambda x: idx2item[x])(preds)
    get_full_sort_score(valid_answers, preds)

    target_users = train_ratings['user_id'].unique()
    user_indices = list(map(lambda u: user2idx[u], target_users))

    target_items = train_ratings['item_id'].unique()
    item_indices = list(map(lambda i: item2idx[i], target_items))
    
    scores = total_pred_rating_mat[user_indices, :][:, item_indices]

    # Web4rec - Experiment Process
    prediction_matrix = pd.DataFrame(
        index=list(map(lambda u: idx2user[u], user_indices)),
        columns=list(map(lambda i: idx2item[i], item_indices)),
        data=scores
    )

    print(prediction_matrix)
    Web4Rec.upload_expermient(
        experiment_name=args.exp_name,
        hyper_parameters={
            'lambda': args.weight_decay
        },
        prediction_matrix=prediction_matrix
    )


if __name__ == '__main__':
    args = get_parser()
    set_seed(args.seed)
    main(args)