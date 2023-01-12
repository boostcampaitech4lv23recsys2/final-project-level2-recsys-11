import os
from copy import deepcopy
import json
import pickle
from tqdm import tqdm

from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk


if __name__ == '__main__':
    model_config_path = os.path.join(os.path.dirname(__file__), 'model_configs') 
    if not os.path.exists(model_config_path):
        os.mkdir(model_config_path)

    model_config = {}

    model_dir_path = os.path.join(os.path.dirname(__file__), 'saved')
    for i, model_file_name in enumerate(tqdm(os.listdir(model_dir_path))):
        model_path = os.path.join(model_dir_path, model_file_name)

        config, model, dataset, train_data, valid_data, test_data = \
            load_data_and_model(model_path)

        model_config['batch_size'] = config['train_batch_size']
        model_config['epochs'] = config['epochs']
        model_config['neg_distribution'] = config['train_neg_sample_args']['distribution']
        model_config['neg_sample_num'] = config['train_neg_sample_args']['sample_num']
        model_config['neg_min_rating'] = int(config['val_interval']['rating'][1])
        model_config['embedding_size'] = config['embedding_size']
        model_config['weight_decay'] = config['weight_decay']

        model_config['user_latent_vector'] = model.user_embedding.weight.detach().cpu().numpy()
        model_config['item_latent_vector'] = model.item_embedding.weight.detach().cpu().numpy()

        user_token2id = deepcopy(dataset.field2token_id[dataset.uid_field])
        model_config['user2idx'] = user_token2id
        item_token2id = deepcopy(dataset.field2token_id[dataset.iid_field])
        model_config['item2idx'] = item_token2id

        
        preds = []
        for user, uid in user_token2id.items():
            if user == '[PAD]': continue

            topk_score, topk_iid_list = \
                full_sort_topk([uid], model=model, test_data=test_data, k=100, device=config['device'])
            external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
            for item in external_item_list[0]:
                preds.append([user, item])

        model_config['topk'] = preds
        with open(os.path.join(model_config_path, f'BPR-{i:03}.pickle', 'wb')) as f:
            pickle.dump(model_config, f)
        # with open(f'BPR-{i:03}.json', 'w') as f:
        #     json.dump(model_config, f, indent=4)

        break


    # user_token2id = deepcopy(dataset.field2token_id[dataset.uid_field])
    # del user_token2id['[PAD]']
    # model_name = args.model_path.split('/')[6][:3]

    # if model_name == 'BPR':
    #     item_vector = model.item_embedding.weight.cpu().detach().numpy()
    #     np.save(f'./saved/{model_name}_itemvector', item_vector)
        
    # id_item_df = pd.DataFrame(list(test_data._dataset.field2token_id['item_id']), columns=['item_id'])
    # id_item_df.to_csv(f'./saved/{model_name}_id_item_df.csv', index=False)
    # print(dataset)
    
    # preds = []
    # for user, uid in tqdm(user_token2id.items()):
    #     topk_score, topk_iid_list = \
    #         full_sort_topk([uid], model=model, test_data=test_data, k=args.topk, device=config['device'])
    #     external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
    #     for item in external_item_list[0]:
    #         preds.append([user, item])