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
from copy import deepcopy

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--topk', default=10, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    config, model, dataset, train_data, valid_data, test_data = \
        load_data_and_model(model_file=args.model_path)
    
    user_token2id = deepcopy(dataset.field2token_id[dataset.uid_field])
    del user_token2id['[PAD]']

    print(dataset)
    
    preds = []
    for user, uid in tqdm(user_token2id.items()):
        topk_score, topk_iid_list = \
            full_sort_topk([uid], model=model, test_data=test_data, k=args.topk, device=config['device'])
        external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
        for item in external_item_list[0]:
            preds.append([user, item])
    
    pd.DataFrame(
        data=preds,
        columns=['user', 'item']
    ).to_csv(f'{config["model"]}_topk.csv', sep='\t', index=False)
