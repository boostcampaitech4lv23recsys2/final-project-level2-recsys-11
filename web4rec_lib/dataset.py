from typing import Union, List, Dict

import pandas as pd
import numpy as np
import requests


class W4RException(Exception):
    pass


def df_to_dict(df: pd.DataFrame):
    ret = {}
    ret['columns'] = df.columns.tolist()
    ret['values'] = df.values.tolist()
    return ret


def dict_to_df(dict_: dict):
    return pd.DataFrame(dict_['values'], columns=dict_['columns'])


class W4RDatasetManager:
    __server_url: str = 'http://127.0.0.1:30004/web4rec-lib' 
    __ID: str = None # user_id -> ID #
    __password: str = None

    dataset_name: str = None #

    user_ids: List = None
    item_ids: List = None

    n_users: int = None
    n_items: int = None

    train_interaction: pd.DataFrame = None #
    ground_truth: pd.DataFrame = None #

    user_side: pd.DataFrame = None #
    item_side: pd.DataFrame = None # + 

    item_popularity: pd.Series = None # {item_id: 'popularity' float } # 영화 본 유저 / 총 유저
    item_name: Dict = {} # {item_id: 'name' str }
    item_vector: Dict = None # 장르스 같은 자카드 item_vecots['genres']


    def login(ID: str, password: str):
        response = requests.get(
            url=W4RDatasetManager.__server_url + '/login',
            params={
                'ID': ID,
                'password': password
            }
        ).json()

        if response[0] == ID:
            print(f'login: {ID} log in success.')
        else:
            raise W4RException(f'login: ID[{ID}] failed.')

        W4RDatasetManager.__ID = ID
        W4RDatasetManager.__password = password


    def add_dataset_name(dataset_name: str):
        if W4RDatasetManager.__ID is None:
            raise W4RException(f'dataset_name: login first')

        # check_dataset
        response = requests.get(
            url=W4RDatasetManager.__server_url + '/check_dataset',
            params={
                'ID': W4RDatasetManager.__ID,
            }
        ).json()

        if dataset_name in response:
            raise W4RException(f'dataset_name: You already have {dataset_name} dataset. change name or remove it.')

        W4RDatasetManager.dataset_name = dataset_name


    def add_train_interaction(train_interaction: pd.DataFrame):
        # train_interaction = train_interaction.copy()
        train_interaction = train_interaction.astype({'user_id': str,'item_id': str})

        # train_interaction['user_id'] = train_interaction['user_id'].astype(str)
        # train_interaction['item_id'] = train_interaction['item_id'].astype(str)

        W4RDatasetManager.user_ids = train_interaction['user_id'].unique().tolist()
        W4RDatasetManager.item_ids = train_interaction['item_id'].unique().tolist()

        W4RDatasetManager.n_users = train_interaction['user_id'].nunique()
        W4RDatasetManager.n_items = train_interaction['item_id'].nunique()

        W4RDatasetManager.train_interaction = train_interaction[['user_id', 'item_id']]


    def add_ground_truth_interaction(ground_truth: pd.DataFrame):
        if W4RDatasetManager.train_interaction is None:
            raise W4RException('ground_truth: You should add train interaction first.')
        
                # train_interaction = train_interaction.copy()
        ground_truth = ground_truth.astype({'user_id': str,'item_id': str})

        # ground_truth['user_id'] = ground_truth['user_id'].astype(str)
        # ground_truth['item_id'] = ground_truth['item_id'].astype(str)

        ground_truth_user_ids = ground_truth['user_id'].unique().tolist()
        ground_truth_item_ids = ground_truth['item_id'].unique().tolist()

        extra_users = set(ground_truth_user_ids) - set(W4RDatasetManager.user_ids)
        if extra_users:
            print(f'ground_truth: {list(extra_users)[:3]}... users are not in train_interaction.')

        extra_items = set(ground_truth_item_ids) - set(W4RDatasetManager.item_ids)
        if extra_items:
            print(f'ground_truth: {list(extra_items)[:3]}... items are not in train_interaction.')
            

        W4RDatasetManager.ground_truth = ground_truth[['user_id', 'item_id']]


    def add_user_side(user_side: pd.DataFrame):
        """
        Should put only cate col.
        pandas category not support
        """
        if W4RDatasetManager.train_interaction is None:
            raise W4RException('user_side: You should add train interaction first.')
        if W4RDatasetManager.ground_truth is None:
            raise W4RException('user_side: You should add ground truth first.')


        user_side = user_side.astype({'user_id': str})

        # 그냥 train user 에 없는 유저는 뺀다.
        user_side_user_ids = user_side['user_id'].unique().tolist()
        extra_users = set(user_side_user_ids) - set(W4RDatasetManager.user_ids)
        if extra_users:
            print(f'user_side: {list(extra_users)[:3]}... users are not in train_interaction. they will be dropped.')
        user_side = user_side[user_side['user_id'].isin(W4RDatasetManager.user_ids)]

        # user_side 를 돌면서 str, int 로만 구성되는지 체크함.
        if user_side.columns.tolist().remove('user_id'):
            raise W4RException(f'user_side: there is no user_side information. (only user_id)')

        for user_col in user_side.columns:
            if user_side[user_col].dtype == float:
                raise W4RException(f'user_side: {user_col} column dtype is not str or int.')

        W4RDatasetManager.user_side = user_side


    def add_item_side(item_side: pd.DataFrame):
        """
        Should put only cate col.
        pandas category not support
        """
        if W4RDatasetManager.train_interaction is None:
            raise W4RException('item_side: You should add train interaction first.')
        if W4RDatasetManager.ground_truth is None:
            raise W4RException('item_side: You should add ground truth first.')

        
        item_side = item_side.astype({'item_id': str})

        item_side_item_ids = item_side['item_id'].unique().tolist()
        extra_items = set(item_side_item_ids) - set(W4RDatasetManager.item_ids)
        if extra_items:
            print(f'item_side: {list(extra_items)[:10]} items are not in train_interaction. they will be dropped.')
        item_side = item_side[item_side['item_id'].isin(W4RDatasetManager.item_ids)]

        # item_side 를 돌면서 str, int 로만 구성되는지 체크함.
        if item_side.columns.tolist().remove('item_id'):
            raise W4RException(f'item_side: there is no item_side information. (only item_id)')

        for item_col in item_side.columns:
            if item_side[item_col].dtype == float:
                raise W4RException(f'item_side: {item_col} column dtype is not str or int.')


        W4RDatasetManager.item_side = item_side


    def add_item_name(item_name: pd.Series):
        if W4RDatasetManager.item_side is None:
            raise W4RException('item_name: You should add item_side first.')

        item_name.index = item_name.index.astype(str)
        W4RDatasetManager.item_side['item_name'] = \
            W4RDatasetManager.item_side['item_id'].map(item_name)


    def add_item_vector(
        item_vector: pd.Series
    ):
        # vector 들의 차원이 동일한지 확인
        # vector 들의 차원이 너무 큰지 확인 ( <= 1024 )
        # if W4RDatasetManager.item_side is None:
        #     raise W4RException('item_vecotr: You should add item_side first.')

        # vector_lens = list(map(lambda x: len(x), item_vector.values))
        # # if np.mean(vector_lens) != len(vector_lens):
        # #     raise W4RException('item_vector: item_vectors values are not same dimension.')

        # if max(vector_lens) > 1024:
        #     raise W4RException('item_vector: W4R supports maximum 1024 dimension vector.')

        item_vector.index = item_vector.index.astype(str)
        W4RDatasetManager.item_side['item_vector'] = W4RDatasetManager.item_side['item_id'].map(item_vector)


    # upload 하면 된다.
    def upload_dataset(desc: str):
        # popularity 구하기
        train = W4RDatasetManager.train_interaction
        item_popularity = \
            train.groupby('item_id')['user_id'].count() / train['user_id'].nunique()
        W4RDatasetManager.item_side['item_popularity'] = item_popularity.values

        body = {
            'ID': W4RDatasetManager.__ID, # str
            'dataset_name': W4RDatasetManager.dataset_name,
            'dataset_desc': desc,
            'train_interaction': W4RDatasetManager.train_interaction.to_dict('tight'), # dict
            'ground_truth': W4RDatasetManager.ground_truth.to_dict('tight'), # dict
            'user_side': W4RDatasetManager.user_side.to_dict('tight'), # dict
            'item_side': W4RDatasetManager.item_side.to_dict('tight'),
        }

        response = requests.post(
            url=W4RDatasetManager.__server_url + '/upload_dataset',
            json=body
        ).json()

        if response['dataset_name'] == body['dataset_name']:
            print(f'ID: {W4RDatasetManager.__ID} Dataset: {W4RDatasetManager.dataset_name} upload complete.')
        else:
            raise W4RException('upload dataset: upload failed.')


if __name__=='__main__':
    # test

    W4RDatasetManager.login(ID='mkdir', password='mkdir')

    W4RDatasetManager.add_dataset_name('ml-1m')

    train_inter = pd.read_csv('/opt/ml/final-project-level2-recsys-11/torch_project/data/ml-1m/train_ratings.csv')
    W4RDatasetManager.add_train_interaction(train_inter)

    ground_truth = pd.read_csv('/opt/ml/final-project-level2-recsys-11/torch_project/data/ml-1m/test_ratings.csv')
    W4RDatasetManager.add_ground_truth_interaction(ground_truth)

    user_side = pd.read_csv('/opt/ml/final-project-level2-recsys-11/torch_project/data/ml-1m/users.csv')
    W4RDatasetManager.add_user_side(user_side[['user_id', 'gender', 'age', 'occupation', 'zip_code_2']])

    item_side = pd.read_csv('/opt/ml/final-project-level2-recsys-11/torch_project/data/ml-1m/items.csv')
    item_side['genres'] = item_side['genres'].apply(lambda s: s.replace('|', ' '))
    item_side.rename(columns= {'genres': 'genres:multi'}, inplace=True)

    W4RDatasetManager.add_item_side(item_side[['item_id', 'year', 'genres:multi']])

    item_name = item_side[['item_id', 'title']].set_index('item_id')['title'].squeeze()
    W4RDatasetManager.add_item_name(item_name)


    item_side['genres_vector'] = item_side['genres:multi'].apply(lambda x: x.split(' '))

    genres = set()
    for _, item_genre in item_side['genres_vector'].iteritems():
        genres = genres | set(item_genre)

    genre2idx = {v:k for k, v in enumerate(genres)}
    item_side['genres_vector'] = item_side['genres_vector'].apply(lambda genres: [genre2idx[genre] for genre in genres])

    item_vector = {}
    print(item_side[['item_id', 'genres_vector']])
    for _, values in item_side[['item_id', 'genres_vector']].iterrows():
        vector = np.zeros(len(genres), dtype=int)
        vector[values['genres_vector']] = 1
        item_vector[values['item_id']] = vector.tolist()
    

    W4RDatasetManager.add_item_vector(pd.Series(item_vector))

    W4RDatasetManager.upload_dataset('movie lens 1 milion.')