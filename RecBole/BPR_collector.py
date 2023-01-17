import os
from copy import deepcopy
import json
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

# TOP N (K < N) 개의 모델 예측값들을 출력

if __name__ == '__main__':
    model_config_path = os.path.join(os.path.dirname(__file__), '..', 'BPR_configs') 
    if not os.path.exists(model_config_path):
        os.mkdir(model_config_path)

    model_config = {}
    model_dir_path = os.path.join(os.path.dirname(__file__), 'saved')
    
    model_list = [path for path in os.listdir(model_dir_path) if 'BPR' in path]
    for i, model_file_name in enumerate(tqdm(model_list)):
        model_path = os.path.join(model_dir_path, model_file_name)

        config, model, dataset, train_data, valid_data, test_data = \
            load_data_and_model(model_path)

        model_config['neg_distribution'] = config['train_neg_sample_args']['distribution']
        model_config['neg_sample_num'] = config['train_neg_sample_args']['sample_num']
        model_config['embedding_size'] = config['embedding_size']
        model_config['weight_decay'] = config['weight_decay']

        # model_config[''] = model.user_embedding.weight.detach().cpu().numpy()
        model_config['ITEM_VECTOR'] = model.item_embedding.weight.detach().cpu().numpy()

        user_token2id = deepcopy(dataset.field2token_id[dataset.uid_field])
        model_config['USER2IDX'] = user_token2id
        item_token2id = deepcopy(dataset.field2token_id[dataset.iid_field])
        model_config['ITEM2IDX'] = item_token2id

        pred_item = {}
        pred_score = {}

        for user, uid in user_token2id.items():
            if user == '[PAD]': continue

            topk_score, topk_iid_list = \
                full_sort_topk([uid], model=model, test_data=test_data, k=30, device=config['device'])
            external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())

            pred_item[user] = np.array(external_item_list[0])
            pred_score[user] = np.array(topk_score.cpu())

        model_config['PRED_ITEM'] = pd.Series(pred_item.values(), index=pred_item.keys())
        model_config['PRED_SCORE'] = pd.Series(pred_score.values(), index=pred_score.keys())

        with open(os.path.join(model_config_path, f'BPR_{i:03}.pickle'), 'wb') as f:
            pickle.dump(model_config, f)
            