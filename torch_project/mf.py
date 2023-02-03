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


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data/ml-1m')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_valids', default=2, type=int)
    
    parser.add_argument("--loss", default='bce', choices = ['bpr', 'bce'], type=str)
    parser.add_argument("--sampler", default='uni', choices=['uni', 'pop'], type=str)
    parser.add_argument("--n_negs", default=2, type=int)

    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--n_epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=4096, type=int)

    # model dependent
    parser.add_argument("--embedding_size", default=8, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=int)

    args = parser.parse_args()
    return args


class TripletDataset(Dataset):
    def __init__(self, negative_sampler: NegativeSampler):
        self.neg_sampler = negative_sampler

        self.triplets = []
        self.sampling()

    def sampling(self):
        self.triplets = []
        neg_samples = self.neg_sampler.sampling()
        user_pos = self.neg_sampler.user_positives
        for u, ps in user_pos.iteritems():
            for p, n in zip(np.repeat(ps, self.neg_sampler.n_negs), neg_samples[u]):
                self.triplets.append((u, p, n))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        return torch.LongTensor(self.triplets[index])


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, embedding_size):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.item_embedding = nn.Embedding(n_items, embedding_size)
    
    def forward(self, u, i):
        user_emb = self.user_embedding(u)
        item_emb = self.item_embedding(i)

        return torch.mul(user_emb, item_emb).sum(dim=1)


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, positive_scores, negative_scores):
        return - torch.sigmoid(positive_scores - negative_scores).log().sum()


def main(args):
    # 라이브러리 쭉쭉
    print('start:', args.loss, args.sampler)
    ratings = pd.read_csv(os.path.join(args.data_dir, 'train_ratings.csv'))
    ratings = ratings.sort_values(by=['user_id', 'timestamp'], ignore_index=True)

    n_users = ratings['user_id'].nunique()
    n_items = ratings['item_id'].nunique()
    
    user2idx = {v: k for k, v in enumerate(ratings['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(ratings['item_id'].unique())}

    idx2user = {k: v for k, v in enumerate(ratings['user_id'].unique())}
    idx2item = {k: v for k, v in enumerate(ratings['item_id'].unique())}

    # save data
    with open('./data/idx2user.pickle','wb') as fw:
        pickle.dump(idx2user, fw)

    with open('./data/idx2item.pickle','wb') as fw:
        pickle.dump(idx2item, fw)

    ratings['user_idx'] = ratings['user_id'].map(user2idx)
    ratings['item_idx'] = ratings['item_id'].map(item2idx)

    # leave n out -> valid answers
    valid_ratings = ratings.groupby('user_id').tail(args.n_valids)
    valid_answers = valid_ratings.groupby('user_id')['item_id'].apply(list).values.tolist()
    train_ratings = ratings.drop(valid_ratings.index)
    

    train_data = train_ratings[['user_idx', 'item_idx']]
    sampler = NegativeSampler(train_data, n_negs=args.n_negs, mode=args.sampler)
    dataset = TripletDataset(sampler)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    model = MatrixFactorization(
        n_users=n_users, 
        n_items=n_items, 
        embedding_size=args.embedding_size
    ).to(args.device)

    criterion = BPRLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.n_epochs):
        epoch_loss = []
        dataset.sampling()
        model.train()
        for batch in tqdm(dataloader, desc=f'[epoch {epoch+1}]'):
            batch = batch.to(args.device)
            users = batch[:, 0]
            pos_items = batch[:, 1]
            neg_items = batch[:, 2]

            positive_scores = model(users, pos_items)
            negative_scores = model(users, neg_items)

            if args.loss == 'bpr':
                loss = criterion(positive_scores, negative_scores)
            else:
                loss1 = nn.BCEWithLogitsLoss()(positive_scores, torch.ones_like(positive_scores).to(args.device))
                loss2 = nn.BCEWithLogitsLoss()(negative_scores, torch.zeros_like(negative_scores).to(args.device))
                loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
        print(f'[epoch {epoch+1}]', 'loss: ', np.mean(epoch_loss))

        model.eval()
        with torch.no_grad():
            user_emb = model.user_embedding.weight.data
            item_emb = model.item_embedding.weight.data
            pred_rating_mat = torch.matmul(user_emb, item_emb.T)
            pred_rating_mat[train_data['user_idx'].values, train_data['item_idx'].values] = -999.9

            _, recs = torch.sort(pred_rating_mat, dim=-1, descending=True)

            preds = recs.cpu().numpy()
            preds = np.vectorize(lambda x: idx2item[x])(preds)
            get_full_sort_score(valid_answers, preds)


    # inference
    topk = {}
    with torch.no_grad():
        user_emb = model.user_embedding.weight.data
        item_emb = model.item_embedding.weight.data

        total_pred_rating_mat = torch.matmul(user_emb, item_emb.T).cpu().numpy()
        # total_pred_rating_mat[train_data['user_idx'].values, train_data['item_idx'].values] = -999.9

        # 필요 유저 - 아이템 들만 골라냄.
        target_users = train_ratings['user_id'].unique()
        user_indices = list(map(lambda u: user2idx[u], target_users))

        target_items = train_ratings['item_id'].unique()
        item_indices = list(map(lambda i: item2idx[i], target_items))

        scores = total_pred_rating_mat[user_indices, :][:, item_indices]

        pd.DataFrame(
            index=list(map(lambda u: idx2user[u], user_indices)),
            columns=list(map(lambda i: idx2item[i], item_indices)),
            data=scores
        ).to_pickle('prediction_matrix_' + args.loss + '_' + args.sampler + '_.pickle')


        # pred_rating_mat_npy = pred_rating_mat.cpu().numpy()
        # np.save('./data/pred_rating_mat.npy', pred_rating_mat_npy)

        # _, recs = torch.sort(pred_rating_mat, dim=-1, descending=True)

        # preds = recs.cpu().numpy()
        # preds = np.vectorize(lambda x: idx2item[x])(preds)
        # for u_idx in range(len(preds)):
        #     topk[idx2user[u_idx]] = preds[u_idx].tolist()
        # # get_full_sort_score(valid_answers, preds)

        # topk = pd.Series(topk)
        # print(topk)


if __name__ == '__main__':
    args = get_parser()
    set_seed(args.seed)
    main(args)