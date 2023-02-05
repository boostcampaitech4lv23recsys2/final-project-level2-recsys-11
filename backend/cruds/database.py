from asyncmy.cursors import DictCursor
from async_lru import alru_cache
from typing import Dict, Tuple, List
import pandas as pd

from database.rds import get_db_inst
from database.s3 import get_from_s3, s3_dict_to_pd


async def insert_from_dict(row: Dict, table: str) -> Tuple:
    placeholders = ", ".join(["%s"] * len(row))
    columns = ", ".join(row.keys())
    query = "INSERT INTO %s ( %s ) VALUES ( %s )" % (table, columns, placeholders)

    return query, tuple(row.values())


# 유저가 등록되있는지 여부 확인
async def check_user(ID: str) -> Dict:
    connection = get_db_inst()

    async with connection as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = "SELECT ID FROM Users WHERE ID = %s"
            await cur.execute(query, (ID,))
            result = await cur.fetchone()
    return result


######## 비동기 객체에 대해 caching이 잘 작동하는지 디버깅 필요!!
@alru_cache(maxsize=5)
async def get_df(ID: str, dataset_name: str):
    connection = get_db_inst()

    async with connection as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = "SELECT * FROM Datasets WHERE ID = %s AND dataset_name = %s"
            await cur.execute(query, (ID, dataset_name))
            row = await cur.fetchone()
    return row


@alru_cache(maxsize=10)
async def get_exp(exp_id: int):
    connection = get_db_inst()

    async with connection as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = "SELECT * FROM Experiments WHERE exp_id = %s"
            await cur.execute(query, (exp_id,))
            row = await cur.fetchone()
    return row


async def inter_to_profile(key_hash: str, group_by: str, col: str) -> pd.DataFrame:
    train_inter = await get_from_s3(key_hash)

    inter_pd = await s3_dict_to_pd(train_inter)
    pd_profile = inter_pd.groupby(group_by).agg(list)[col].reset_index()

    return pd_profile


@alru_cache(maxsize=5)
async def get_total_info(ID: str, dataset_name: str):
    connection = get_db_inst()

    async with connection as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = "SELECT exp_id, experiment_name, alpha, objective_fn, hyperparameters, \
                     recall, map, ndcg, tail_percentage, avg_popularity, coverage, \
                     diversity_cos, diversity_jac, serendipity_pmi, serendipity_jac, novelty, \
                     metric_per_user FROM Experiments WHERE ID = %s AND dataset_name = %s AND alpha = %s"
            await cur.execute(query, (ID, dataset_name, 1))
            result = await cur.fetchall()

    return result


@alru_cache(maxsize=5)
async def get_total_reranked(ID:str, dataset_name:str, exp_names:Tuple[str]):
    placeholders = ", ".join(["%s"] * len(exp_names))

    connection = get_db_inst()
    
    async with connection as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = "SELECT exp_id, experiment_name, alpha, objective_fn, hyperparameters, \
                    recall, map, ndcg, tail_percentage, avg_popularity, coverage, \
                    diversity_cos, diversity_jac, serendipity_pmi, serendipity_jac, novelty, \
                    metric_per_user FROM Experiments \
                    WHERE ID = %s AND dataset_name = %s \
                    AND alpha != %s AND experiment_name IN ({})".format(placeholders)
            await cur.execute(query, (ID, dataset_name, 1, *exp_names))
            result = await cur.fetchall()
            
    return result


async def check_password(ID: str, password: str) -> Dict:
    conn2 = get_db_inst()

    async with conn2 as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = "SELECT ID, password FROM Users WHERE ID = %s AND password = %s"
            await cur.execute(query, (ID, password))
            result = await cur.fetchone()
    return result
