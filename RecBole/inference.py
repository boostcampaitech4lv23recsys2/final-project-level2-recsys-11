from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_model, get_trainer
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

import argparse
import pandas as pd
import numpy as np
from copy import deepcopy

from tqdm import tqdm
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/opt/ml/final-project-level2-recsys-11/RecBole/saved/BPR-Jan-10-2023_02-24-30.pth', type=str) #, required=True
    parser.add_argument('--topk', default=10, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    config, model, dataset, train_data, valid_data, test_data = \
        load_data_and_model(model_file=args.model_path)

    user_token2id = deepcopy(dataset.field2token_id[dataset.uid_field])
    del user_token2id['[PAD]']
    model_name = args.model_path.split('/')[6][:3]

    if model_name == 'BPR':
        item_vector = model.item_embedding.weight.cpu().detach().numpy()
        np.save(f'./saved/{model_name}_itemvector', item_vector)
        
    id_item_df = pd.DataFrame(list(test_data._dataset.field2token_id['item_id']), columns=['item_id'])
    id_item_df.to_csv(f'./saved/{model_name}_id_item_df.csv', index=False)
    print(dataset)
    
    preds = []
    for user, uid in tqdm(user_token2id.items()):
        topk_score, topk_iid_list = \
            full_sort_topk([uid], model=model, test_data=test_data, k=args.topk, device=config['device'])
        external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
        for item in external_item_list[0]:
            preds.append([user, item])
    
    if not os.path.exists(f'inference/'):
        os.makedirs('inference/')
        print("inference folder created")

    pd.DataFrame(
        data=preds,
        columns=['user', 'item']
    ).to_csv(os.path.join('inference', f'{config["model"]}_{config["config_id"]}.csv'), sep='\t', index=False)
