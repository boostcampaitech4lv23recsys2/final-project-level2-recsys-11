from asyncmy.cursors import DictCursor
from async_lru import alru_cache
from typing import Dict, Tuple
from routers.database import get_db_inst


async def insert_from_dict(row: Dict, table: str) -> Tuple:
    placeholders = ', '.join(['%s'] * len(row))
    columns = ', '.join(row.keys())
    query = "INSERT INTO %s ( %s ) VALUES ( %s )" % (table, columns, placeholders)

    return query, tuple(row.values())


# 유저가 등록되있는지 여부 확인
async def check_user(ID:str) -> Dict:
    connection = get_db_inst()
    
    async with connection as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = "SELECT ID FROM Users WHERE ID = %s"
            await cur.execute(query, (ID,))
            result = await cur.fetchone()
    return result 

######## 비동기 객체에 대해 caching이 잘 작동하는지 디버깅 필요!!


@alru_cache(max_size=5)
async def get_df(ID: str, dataset_name: str):
    connection = get_db_inst()

    async with connection as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = "SELECT * FROM Datasets WHERE ID = %s AND dataset_name = %s"
            await cur.execute(query, (ID, dataset_name))
            row = await cur.fetchone()
    return row 


@alru_cache(max_size=5)
async def get_exp(exp_id: int):
    connection = get_db_inst()
    async with connection as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = "SELECT * FROM Experiments WHERE exp_id = %s"
            await cur.execute(query, (exp_id,))
            row = await cur.fetchone()
    return row 
    