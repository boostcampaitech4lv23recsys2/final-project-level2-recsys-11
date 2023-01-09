from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_model, get_trainer

from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
)

import argparse
import pandas as pd
import os
from copy import deepcopy

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='BPR', type=str)
    parser.add_argument('--dataset', default='ml-1m', type=str)
    # parser.add_argument('--yamls', default='test.yaml', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    # configurations initialization
    config = Config(
        dataset=args.dataset,
        config_file_list=['environment.yaml', args.model + '.yaml']
    )

    # # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # # logger initialization
    init_logger(config)
    logger = getLogger()

    # # write config info into log
    logger.info(config)

    # # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # test_data -> 그라운드 트루쓰 로 저장함. dataset.test
    # 훈련마다 저장하게 된다.
    test_tensor = test_data.dataset.inter_feat.interaction
    test_user_id2token = test_data.dataset.field2id_token[config['USER_ID_FIELD']]
    test_item_id2token = test_data.dataset.field2id_token[config['ITEM_ID_FIELD']]
    test_user_ids = test_tensor[config['USER_ID_FIELD']]
    test_item_ids = test_tensor[config['ITEM_ID_FIELD']]
    test_user_tokens = [test_user_id2token[idx] for idx in test_user_ids]
    test_item_tokens = [test_item_id2token[idx] for idx in test_item_ids]
    pd.DataFrame(
        [[u, i] for u, i in zip(test_user_tokens, test_item_tokens)],
        columns=['user', 'item']
    ).to_csv(os.path.join(config['data_path'], config['dataset'] + '.test'), sep='\t', index=False)


    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")


    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    
    print(test_result)