from datetime import datetime
from fastapi import APIRouter, Depends, File, UploadFile
import pandas as pd
from typing import List, Dict

from schemas.data import Dataset, Experiment
from routers.database import get_db

router = APIRouter()

DATASETS = {} # 유저별 데이터셋 in-memory에 따로 저장


# 유저가 등록되있는지 여부 확인
async def check_user(user_id:str, password:str, connection) -> Dict:
    async with connection as conn:
        async with conn.cursor() as cur:
            query = "SELECT user_id FROM User WHERE user_id = %s AND user_pwd = %s"
            await cur.execute(query, (user_id, password))
            result = await cur.fetchone()
            return result


@router.get("/login")
async def login(user_id:str, password:str, connection=Depends(get_db)) -> str:
    user = await check_user(user_id, password, connection)

    # 데이터에서 유저 확인
    if user:
        return list(user)
    else:
        return None


@router.post("/add_user")
async def add_user(user_id: str, password:str, connection=Depends(get_db)):
    async with connection as conn:
        async with conn.cursor() as cur:
            curr_time = await datetime.now()
            query = "INSERT INTO User (user_id, user_pwd, access_time) VALUES (%s, %s, %s)"
            await cur.execute(query, (user_id, password, curr_time))
        await conn.commit()


@router.get("/check_dataset")
async def check_dataset(user_id: str, connection=Depends(get_db)) -> List:
    async with connection as conn:
        async with conn.cursor() as cur:
            query = 'SELECT dataset_name FROM Datasets WHERE user_id = %s'
            await cur.execute(query, (user_id,))
            result = await cur.fetchall() 

    # [row['dataset_name'] for row in result]
    return list(result)


# https://stackoverflow.com/questions/63048825/how-to-upload-file-using-fastapi/70657621#70657621
# https://stackoverflow.com/questions/65504438/how-to-add-both-file-and-json-body-in-a-fastapi-post-request/70640522#70640522

async def check_dataset(user_id:str, dataset_name:str, connection) -> Dict:
    async with connection as conn:
        async with conn.cursor() as cur:
            query = 'SELECT EXISTS(SELECT * FROM Datasets WHERE user_id = %s and dataset_name = %s'
            await cur.execute(query, (user_id, dataset_name))
            result = await cur.fetchone()
            return result


@router.post("/upload_dataset")
async def upload_dataset(user_id: str, dataset_name:str, 
                         dataset: Dataset,
                         connection = Depends(get_db)) -> Dataset:

    ### TODO: json (library) -> json (s3) 

    # s3_locations-> (user_id, dataset_name, train_df, ground_truth, user_side, item_side, item2idx, user2idx)
    # S3 Object Name = 고유 키 값

    async with connection as conn:
        async with conn.cursor() as cur:
            
            # 고유값이 중복인 dataset 제거
            if check_dataset(user_id, dataset_name, connection):
                query_dataset_del = 'DELETE FROM Datasets WHERE user_id = %s and dataset_name = %s'
                query_exp_del = 'DELETE FROM Experiments WHERE user_id = %s and dataset_name = %s'
                cur.execute(query_dataset_del, (user_id, dataset_name))
                cur.execute(query_exp_del, (user_id, dataset_name))

            # 주어진 dataset 추가
            query_dataset_ins = 'INSERT INTO Datasets (user_id, dataset_name, train_df, ground_truth, \
                    user_side, item_side) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)' 
            cur.execute(query_dataset_ins, (user_id, dataset_name, S3_train_df, ...)) # S3 object 이름 필요
        
        await conn.commit()

    DATASETS[user_id] = dataset 

    return dataset # library 쪽으로 return (필요 없을수도)


async def get_metrics():
    pass


async def upload_experiment(user_id: str, dataset_name:str,
                            experiment: Experiment,
                            connection=Depends(get_db)) -> Experiment:
    
    ### TODO: experiment -> s3, 주소 변수로 저장
    ### TODO: 지표 계산 (dataset: Dataset 필요)

    async with connection as conn:
        async with conn.cursor() as cur:
            query = 'INSERT INTO Experiment (user_id, dataset_name, experiment_name, alpha, objective_fn, ...'
            cur.exectue(query, (user_id, dataset_name, ...)) # S3 object 이름 필요

        await conn.commit()