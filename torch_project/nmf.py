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


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data/ml-1m')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_valids', default=2, type=int)
    
    
    parser.add_argument("--exp_name", type=str, default='NueMF')

    parser.add_argument("--loss", default='bpr', choices = ['bpr', 'bce'], type=str)
    parser.add_argument("--sampler", default='pop', choices=['uni', 'pop'], type=str)
    parser.add_argument("--n_negs", default=2, type=int)

    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=4096, type=int)

    # model dependent
    parser.add_argument("--gmf_embedding_size", default=32, type=int)
    parser.add_argument("--mlp_embedding_size", default=32, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument('--layers', nargs='+', default=[64, 32, 8])

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


class NueMF(nn.Module):
    def __init__(self, n_users, n_items, gmf_embedding_size, mlp_embedding_size, layers, dropout = 0.2):
        super().__init__()
        self.args = args

        self.n_users = n_users
        self.n_items = n_items

        # GMF
        self.GMF_user_embedding = nn.Embedding(n_users, gmf_embedding_size)
        self.GMF_item_embedding = nn.Embedding(n_items, gmf_embedding_size)
        # self.GMF_fc = nn.Linear(gmf_embedding_size, 1)

        # MLP
        self.MLP_user_embedding = nn.Embedding(n_users, mlp_embedding_size)
        self.MLP_item_embedding = nn.Embedding(n_items, mlp_embedding_size)

        mlp_layers = [
            nn.Linear(mlp_embedding_size * 2, layers[0]), 
            nn.ReLU()
        ]

        for i in range(len(layers) - 1):
            mlp_layers.append(nn.Dropout(dropout))
            mlp_layers.append(nn.Linear(layers[i], layers[i+1]))
            mlp_layers.append(nn.ReLU())
        
        self.MLP_layers = nn.Sequential(*mlp_layers)
        # self.MLP_fc = nn.Linear(mlp_layers[-1], 1)
        
        # NeuMF
        self.NMF_fc = nn.Linear(gmf_embedding_size + layers[-1], 1)


    def forward(self, user, item):
        gmf_user_emb = self.GMF_user_embedding(user)
        gmf_item_emb = self.GMF_item_embedding(item)
        gmf_out = torch.mul(gmf_user_emb, gmf_item_emb) # (B, gmf_f)

        mlp_user_emb = self.MLP_user_embedding(user)
        mlp_item_emb = self.MLP_item_embedding(item)
        mlp_out = self.MLP_layers(torch.cat([mlp_user_emb, mlp_item_emb], dim=1)) # (B, mlp_f)

        return self.NMF_fc(torch.cat([gmf_out, mlp_out], dim=1)).squeeze()


class TotalTestDataset(Dataset):
    def __init__(self, n_users, n_items):
        self.n_users = n_users,
        self.n_items = n_items,
        self.users = torch.arange(n_users)
        self.items = torch.arange(n_items)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return torch.stack([self.users[index].repeat(self.n_items), self.items], dim=1)


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, positive_scores, negative_scores):
        return - torch.sigmoid(positive_scores - negative_scores).log().sum()


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
    print(train_data['item_idx'].max(), train_data['item_idx'].nunique())
    sampler = NegativeSampler(train_data, n_negs=args.n_negs, mode=args.sampler)
    dataset = TripletDataset(sampler)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    

    test_dataset = TotalTestDataset(n_users, n_items)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    model = NueMF(
        n_users=n_users, 
        n_items=n_items,
        gmf_embedding_size=args.gmf_embedding_size,
        mlp_embedding_size=args.mlp_embedding_size,
        layers=args.layers,
        dropout=args.dropout
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
            pred_rating_mat = torch.empty((n_users, n_items))
            for i, batch in tqdm(enumerate(test_dataloader), total=n_users):
                batch = batch.to(args.device)
                batch = batch.squeeze(0)
                pred_rating_mat[i][...] = model(batch[:, 0], batch[:, 1])
                
            pred_rating_mat[train_data['user_idx'].values, train_data['item_idx'].values] = -999.9

            _, recs = torch.sort(pred_rating_mat, dim=-1, descending=True)

            preds = recs.cpu().numpy()
            preds = np.vectorize(lambda x: idx2item[x])(preds)
            get_full_sort_score(valid_answers, preds)


    # inference
    model.eval()
    with torch.no_grad():
        pred_rating_mat = torch.empty((n_users, n_items))
        for i, batch in tqdm(enumerate(test_dataloader), total=n_users):
            batch = batch.to(args.device)
            batch = batch.squeeze(0)
            pred_rating_mat[i][...] = model(batch[:, 0], batch[:, 1])

        target_users = train_ratings['user_id'].unique()
        user_indices = list(map(lambda u: user2idx[u], target_users))

        target_items = train_ratings['item_id'].unique()
        item_indices = list(map(lambda i: item2idx[i], target_items))
        
        scores = pred_rating_mat[user_indices, :][:, item_indices]

        # Web4rec - Experiment Process
        prediction_matrix = pd.DataFrame(
            index=list(map(lambda u: idx2user[u], user_indices)),
            columns=list(map(lambda i: idx2item[i], item_indices)),
            data=scores
        )

        Web4Rec.upload_expermient(
            experiment_name=args.exp_name,
            hyper_parameters={
                'sampling': args.sampler,
                'gmf_embedding_size': args.gmf_embedding_size,
                'mlp_layers': args.layers
            },
            prediction_matrix=prediction_matrix
        )
        # Web4rec - Experiment Process end


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