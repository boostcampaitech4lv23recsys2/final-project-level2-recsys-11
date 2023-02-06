import os
import pickle
import time
from typing import Union, List, Dict
import json

import requests
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, linalg

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE,Isomap

from .src.util import Web4RecDataset, W4RException
from .src.metric import get_total_information
from .src.distance import get_distance_mat, get_jaccard_mat
from .src.rerank import get_total_reranks


np.set_printoptions(suppress=True, precision=3)


class Web4Rec:
    # for log in
    __server_url: str = 'http://127.0.0.1:30004/web4rec-lib' 
    __token: str = None # user_id -> ID #
    __ID: str = None

    dataset: Web4RecDataset = None
    # __password: str = None


    def login(token: str):
            response = requests.get(
                url=Web4Rec.__server_url + '/login',
                params={
                    'token': token
                }
            ).json()

            if response is None:
                raise W4RException('token error.')

            print(f"[W4R] login: ID[{response['ID']}] log in success.")
            Web4Rec.__token = token
            Web4Rec.__ID = response['ID']

        
    def register_dataset(dataset: Web4RecDataset):
        # check dataset
        response = requests.get(
            Web4Rec.__server_url + '/check_datasets',
            params={
                'token': Web4Rec.__token
            }
        ).json()

        if dataset.dataset_name not in [dataset['dataset_name'] for dataset in response]:
            print(f"[W4R] upload_dataset: '{dataset.dataset_name}' doesn't exist in your account. uploading..")
        
            body = {
                'ID': Web4Rec.__ID, # str
                'dataset_name': dataset.dataset_name,
                'dataset_desc': dataset.dataset_desc,
                'train_interaction': dataset.train_interaction.to_dict('tight'), # dict
                'ground_truth': dataset.ground_truth.to_dict('tight'), # dict
                'user_side': dataset.user_side.to_dict('tight'), # dict
                'item_side': dataset.item_side.to_dict('tight'),
            }

            response = requests.post(
                url=Web4Rec.__server_url + '/upload_dataset',
                json=body
            ).json()

            print(f'[W4R] upload_dataset: time: {response} upload complete.')

        else:
            print(f'[W4R] upload_dataset: {dataset.dataset_name} exists.')

        Web4Rec.dataset = dataset


    def upload_expermient(
        experiment_name: str,
        hyper_parameters: Dict[str, str],
        prediction_matrix: pd.DataFrame = None,
        n_candidates=100,
        k=10,
    ):
        print(f'[W4R] upload experiment...')
        if Web4Rec.dataset is None:
            raise W4RException('[W4R] upload_experiment: You should register Web4RecDataset object first.')

        experiment_time = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))

        # prediction user_id, item_id type to str
        prediction_matrix.index = prediction_matrix.index.astype(str)
        prediction_matrix.columns = prediction_matrix.columns.astype(str)

        # prediction matrix scaling!
        mins = prediction_matrix.min(axis=1)
        nominator = prediction_matrix.sub(mins, axis='index')
        scaled_prediction_matrix = nominator.div(prediction_matrix.max(axis=1) - mins, axis='index')
        prediction_matrix = scaled_prediction_matrix

        user_tsne_df, item_tsne_df = Web4Rec.__get_tsne(prediction_matrix)


        # encoding
        pred_users = prediction_matrix.index.tolist()
        pred_items = prediction_matrix.columns.tolist()
        pred_mat = prediction_matrix.values


        uid2idx = {v: k for k, v in enumerate(pred_users)}
        iid2idx = {v: k for k, v in enumerate(pred_items)}

        idx2uid = {k: v for k, v in enumerate(pred_users)}
        idx2iid = {k: v for k, v in enumerate(pred_items)}

        train_interaction = Web4Rec.dataset.train_interaction
        train_interaction = train_interaction[train_interaction['user_id'].isin(pred_users)]
        train_interaction = train_interaction[train_interaction['item_id'].isin(pred_items)]
        train_interaction['user_idx'] = train_interaction['user_id'].map(uid2idx)
        train_interaction['item_idx'] = train_interaction['item_id'].map(iid2idx)


        # distance matrix ready
        cos_dist, pmi_dist = get_distance_mat(train_interaction)
        jac_dist = None
        if 'item_vector' in Web4Rec.dataset.item_side.columns:
            item_vector = Web4Rec.dataset.item_side[['item_id', 'item_vector']].set_index('item_id').squeeze()
            item_vector.index = item_vector.index.map(iid2idx)
            jac_dist = get_jaccard_mat(item_vector)


        # quant prepare ready
        user_profile = train_interaction.groupby('user_idx')['item_idx'].apply(list)
        item_popularity = \
            train_interaction.groupby('item_idx')['user_idx'].count() / train_interaction['user_idx'].nunique()
        tail_items = item_popularity.index[-int(len(item_popularity) * 0.8):].tolist()
        total_items = train_interaction['item_idx'].unique() 


        # already seen item to 0
        pred_mat[train_interaction['user_idx'], train_interaction['item_idx']] = 0.

        # ground truth encoding
        ground_truth = Web4Rec.dataset.ground_truth
        ground_truth = ground_truth[ground_truth['user_id'].isin(pred_users)]
        ground_truth = ground_truth[ground_truth['item_id'].isin(pred_items)]

        ground_truth['user_idx'] = ground_truth['user_id'].map(uid2idx)
        ground_truth['item_idx'] = ground_truth['item_id'].map(iid2idx)

        actuals = list(ground_truth.groupby('user_idx')['item_idx'].apply(list))

        # now make json body base experiment and reranks
        candidates = np.argsort(-pred_mat, axis=1)[:, :n_candidates]

        # base model
        predicts = candidates[:, :n_candidates] # candidates 로 넣는다

        metrices, metric_per_user = get_total_information(
            predicts=predicts,
            actuals=actuals,
            cos_dist=cos_dist,
            pmi_dist=pmi_dist,
            user_profile=user_profile,
            item_popularity=item_popularity,
            tail_items=tail_items,
            total_items=total_items,
            jac_dist=jac_dist,
            k=k
        )

        # base predicts decode
        decode_predicts = np.vectorize(lambda x: idx2iid[x])(predicts)
        decode_pred_items = pd.Series({i: v.tolist() for i, v in enumerate(decode_predicts)})
        decode_pred_items.index = decode_pred_items.index.map(idx2uid)
        decode_pred_items_df = pd.DataFrame(decode_pred_items, columns=['pred_items'])
        
        # base predicts score decode
        decode_pred_scores = {}
        for u, predicted in decode_pred_items.iteritems():
            decode_pred_scores[u] = [float(prediction_matrix.loc[u, pred]) for pred in predicted]
        decode_pred_scores = pd.Series(decode_pred_scores)
        decode_pred_scores_df = pd.DataFrame(decode_pred_scores, columns=['pred_scores'])

        
        decode_pred_items_df = decode_pred_items_df.reset_index()
        decode_pred_items_df = decode_pred_items_df.rename(columns = {'index': 'user_id'})
        
        decode_pred_scores_df = decode_pred_scores_df.reset_index()
        decode_pred_scores_df = decode_pred_scores_df.rename(columns = {'index': 'user_id'})

        # print(decode_pred_items_df)
        # print(decode_pred_scores_df)

        # print(metric_per_user)
        body = {
            'ID': Web4Rec.__ID, # str
            'dataset_name': Web4Rec.dataset.dataset_name,
            'experiment_name': experiment_name + '_' + experiment_time,
            'alpha': 1,
            'objective_fn': None,

            'hyperparameters': json.dumps(hyper_parameters),
            'pred_items': decode_pred_items_df.to_dict(orient='tight'),
            'pred_scores': decode_pred_scores_df.to_dict(orient='tight'),

            'user_tsne': user_tsne_df.to_dict(orient='tight'),
            'item_tsne': item_tsne_df.to_dict(orient='tight'),

            'recall': metrices['recall'],
            'ndcg': metrices['ndcg'],
            'map': metrices['avg_precision'],
            'avg_popularity': metrices['avg_popularity'],
            'tail_percentage': metrices['tail_percentage'],
            'coverage': metrices['coverage'],

            'diversity_cos': metrices['diversity_cos'],
            'serendipity_pmi': metrices['serendipity_pmi'],
            'novelty': metrices['novelty'],

            'metric_per_user': pd.DataFrame(metric_per_user, columns=['metric_per_used']).to_dict(orient='tight')
        }

        if jac_dist is not None:
            body['diversity_jac'] = metrices['diversity_jac']
            body['serendipity_jac'] = metrices['serendipity_jac']

        response = requests.post(
            url=Web4Rec.__server_url + '/upload_experiment',
            json=body
        ).json()

        # reranks -
        print('[W4R] upload experiment: reranking... ')
        for rerank_name in ['diversity(cos)', 'serendipity(pmi)', 'novelty', 'diversity(jac)', 'serendipity(jac)']:
            if rerank_name in ['diversity(jac)', 'serendipity(jac)']:
                if jac_dist is None:
                    continue
            print('[W4R]', rerank_name, flush=True)
            if 'cos' in rerank_name:
                dist_mat = cos_dist
            elif 'pmi' in rerank_name:
                dist_mat = pmi_dist
            else:
                dist_mat = jac_dist
            
            predicts = get_total_reranks(
                mode=rerank_name,
                candidates=candidates,
                prediction_mat=pred_mat,
                distance_mat=dist_mat,
                user_profile=user_profile,
                item_popularity=item_popularity,
                alpha=0.5,
                k=k
            )

            metrices, metric_per_user = get_total_information(
                predicts=predicts,
                actuals=actuals,
                cos_dist=cos_dist,
                pmi_dist=pmi_dist,
                user_profile=user_profile,
                item_popularity=item_popularity,
                tail_items=tail_items,
                total_items=total_items,
                jac_dist=jac_dist,
                k=k
            )

            # rerank predicts decode
            decode_predicts = np.vectorize(lambda x: idx2iid[x])(predicts)
            decode_pred_items = pd.Series({i: v.tolist() for i, v in enumerate(decode_predicts)})
            decode_pred_items.index = decode_pred_items.index.map(idx2uid)
            decode_pred_items_df = pd.DataFrame(decode_pred_items, columns=['pred_items'])
        
            decode_pred_items_df = decode_pred_items_df.reset_index()
            decode_pred_items_df = decode_pred_items_df.rename(columns = {'index': 'user_id'})


            body = {
                'ID': Web4Rec.__ID, # str
                'dataset_name': Web4Rec.dataset.dataset_name,
                'experiment_name': experiment_name + '_' + experiment_time,
                'alpha': 0.5,
                'objective_fn': rerank_name,
                'hyperparameters': json.dumps(hyper_parameters),
                'pred_items': decode_pred_items_df.to_dict(orient='tight'),

                'recall': metrices['recall'],
                'ndcg': metrices['ndcg'],
                'map': metrices['avg_precision'],
                'avg_popularity': metrices['avg_popularity'],
                'tail_percentage': metrices['tail_percentage'],
                'coverage': metrices['coverage'],

                'diversity_cos': metrices['diversity_cos'],
                'serendipity_pmi': metrices['serendipity_pmi'],
                'novelty': metrices['novelty'],

                'metric_per_user': pd.DataFrame(metric_per_user, columns=['metric_per_used']).to_dict(orient='tight')
            }

            if jac_dist is not None:
                body['diversity_jac'] = metrices['diversity_jac']
                body['serendipity_jac'] = metrices['serendipity_jac']

            response = requests.post(
                url=Web4Rec.__server_url + '/upload_experiment',
                json=body
            ).json()


    
    def __get_tsne(prediction_matrix: pd.DataFrame, dim=64):
        clf = TruncatedSVD(dim)

        user_pca = clf.fit_transform(prediction_matrix)
        item_pca = clf.fit_transform(prediction_matrix.T)

        model = TSNE() # learning_rate=100
        user_transformed = model.fit_transform(user_pca)

        model = TSNE() # learning_rate=100
        item_transformed = model.fit_transform(item_pca)

        user_tsne = pd.DataFrame(data=user_transformed, columns=['xs', 'ys'], index=prediction_matrix.index)
        item_tsne = pd.DataFrame(data=item_transformed, columns=['xs', 'ys'], index=prediction_matrix.columns)

        user_tsne = user_tsne.reset_index()
        user_tsne = user_tsne.rename(columns = {'index': 'user_id'})

        item_tsne = item_tsne.reset_index()
        item_tsne = item_tsne.rename(columns = {'index': 'item_id'})

        return user_tsne, item_tsne
        
            


if __name__ == '__main__':
    Web4Rec.login(token='mkdir_token')

    w4r_dataset = Web4RecDataset(dataset_name='ml-1m')

    train_interaction = pd.read_csv('/opt/ml/final-project-level2-recsys-11/torch_project/data/ml-1m/train_ratings.csv')
    ground_truth = pd.read_csv('/opt/ml/final-project-level2-recsys-11/torch_project/data/ml-1m/test_ratings.csv')
    user_side = pd.read_csv('/opt/ml/final-project-level2-recsys-11/torch_project/data/ml-1m/users.csv')
    item_side = pd.read_csv('/opt/ml/final-project-level2-recsys-11/torch_project/data/ml-1m/items.csv')
    item_side['genres'] = item_side['genres'].apply(lambda s: s.replace('|', ' '))
    item_side.rename(
        columns= {
            'title': 'item_name',
            'genres': 'genres:multi',
            }, 
        inplace=True
    )
    

    w4r_dataset = Web4RecDataset(dataset_name='ml-1m')
    w4r_dataset.add_train_interaction(train_interaction)
    w4r_dataset.add_ground_truth(ground_truth)
    w4r_dataset.add_user_side(user_side)
    w4r_dataset.add_item_side(item_side)

    Web4Rec.register_dataset(w4r_dataset)

    Web4Rec.upload_expermient(
        experiment_name='MatrixFactorization',
        hyper_parameters={
            'loss': 'bce',
            'sampling': 'popularity'
        },
        prediction_matrix = pd.read_pickle(filepath_or_buffer='/opt/ml/final-project-level2-recsys-11/torch_project/web4rec/asset/prediction_matrix_bce_pop_.pickle')
    )
