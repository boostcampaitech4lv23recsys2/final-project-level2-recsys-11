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

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
parser.add_argument(
    "--dataset", "-d", type=str, default="ml-1m", help="name of datasets"
)
parser.add_argument("--config_files", type=str, default=None, help="config files")
parser.add_argument(
    "--nproc", type=int, default=1, help="the number of process in this group"
)
parser.add_argument(
    "--ip", type=str, default="localhost", help="the ip of master node"
)
parser.add_argument(
    "--port", type=str, default="5678", help="the port of master node"
)
parser.add_argument(
    "--world_size", type=int, default=-1, help="total number of jobs"
)
parser.add_argument(
    "--group_offset",
    type=int,
    default=0,
    help="the global rank offset of this group",
)

args, _ = parser.parse_known_args()

config_file_list = (
    args.config_files.strip().split(" ") if args.config_files else None
)

config = Config(
    model=args.model,
    dataset=args.dataset,
    config_file_list=config_file_list,
)

# 타임스탬프 전처리
dataset_path = os.path.join('/opt/ml/level3_productserving-level3-recsys-11/dataset/', config['dataset'])
config['data_path'] = dataset_path

if os.path.exists(dataset_path):
    raise Exception(f'{config["dataset"]} is already exists. plz remove dataset file and rerun. -hoon')

try:
    dataset = create_dataset(config)
except:
    pass

inter_df = pd.read_csv(
    os.path.join(dataset_path, f'{config["dataset"]}.inter'),
    sep = config['field_separator']
)
inter_df = inter_df.sort_values(
    by=[config['USER_ID_FIELD'] + ':token', config['TIME_FIELD'] + ':float'],
    ignore_index=True
)
inter_df['origin_' + config['TIME_FIELD'] + ':float'] = \
    inter_df[config['TIME_FIELD'] + ':float']
inter_df[config['TIME_FIELD'] + ':float'] = [i for i in range(len(inter_df))]

inter_df.to_csv(
    os.path.join(dataset_path, f'{config["dataset"]}.inter'), 
    sep=config['field_separator'], 
    index=False
)
