from tqdm import tqdm
import random
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, linalg
from utils import NegativeSampler, set_seed, get_full_sort_score


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/opt/ml/final-project-level2-recsys-11/torch/data/ml-1m')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_valids', default=1, type=int)

    # model dependent
    parser.add_argument("--k", default=5, type=int)

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
    
    ratings['user_idx'] = ratings['user_id'].map(user2idx)
    ratings['item_idx'] = ratings['item_id'].map(item2idx)

    # leave n out -> valid answers
    valid_ratings = ratings.groupby('user_id').tail(args.n_valids)
    valid_answers = valid_ratings.groupby('user_id')['item_id'].apply(list).values.tolist()
    train_ratings = ratings.drop(valid_ratings.index)

    train_mat = csr_matrix((np.ones(len(train_ratings)), (train_ratings['user_idx'], train_ratings['item_idx'])))
    item_mat = (train_mat.T @ train_mat).toarray()


if __name__ == '__main__':
    args = get_parser()
    set_seed(args.seed)
    main(args)