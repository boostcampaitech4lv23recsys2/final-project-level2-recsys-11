from datetime import datetime
from fastapi import APIRouter, Depends, File, UploadFile
import pandas as pd
from typing import List, Dict

from schemas.data import Dataset, Experiment
from routers.database import get_db, send_to_s3, get_from_s3, s3_transmission, insert_from_dict

router = APIRouter()

DATASETS = {} # 유저별 데이터셋 in-memory에 따로 저장


# 유저가 등록되있는지 여부 확인
async def check_user(user_id:str, password:str, connection) -> Dict:
    async with connection as conn:
        async with conn.cursor() as cur:
            query = "SELECT user_id FROM Users WHERE user_id = %s AND user_password = %s"
            await cur.execute(query, (user_id, password))
            result = await cur.fetchone()
            return result


@router.get("/login")
async def login(user_id:str, password:str, connection=Depends(get_db)) -> str:
    user = await check_user(user_id, password, connection) 

    # TODO;
    # access time (insert)

    # 데이터에서 유저 확인
    if user:
        return list(user)
    else:
        return None


@router.post("/add_user")
async def add_user(user_id: str, password:str, connection=Depends(get_db)):
    async with connection as conn:
        async with conn.cursor() as cur:
            curr_time = datetime.now()
            query = "INSERT INTO Users (user_id, user_password, access_time) VALUES (%s, %s, %s)"
            await cur.execute(query, (user_id, password, curr_time))
        await conn.commit()


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

async def check_dataset(user_id:str, dataset_name:str, connection) -> bool:
    async with connection as conn:
        async with conn.cursor() as cur:
            query = 'SELECT EXISTS(SELECT * FROM Datasets WHERE user_id = %s and dataset_name = %s'
            await cur.execute(query, (user_id, dataset_name))
            result = await cur.fetchone()
            return result


@router.post("/upload_dataset")
async def upload_dataset(dataset: Dataset,
                         connection = Depends(get_db)) -> Dataset:

    primary_key = dataset.user_id + dataset.dataset_name # str 
    row_dict = s3_transmission(dataset, primary_key)
    row_dict['user_id'] = dataset.user_id
    row_dict['dataset_name'] = dataset.dataset_name

    async with connection as conn:
        async with conn.cursor() as cur:

            if not check_dataset(dataset.user_id, dataset.dataset_name, connection):
                # 고유값이 중복인 dataset 제거 - if check_dataset
                # query_dataset_del = 'DELETE FROM Datasets WHERE user_id = %s and dataset_name = %s'
                # query_exp_del = 'DELETE FROM Experiments WHERE user_id = %s and dataset_name = %s'
                # cur.execute(query_dataset_del, (dataset.user_id, dataset.dataset_name))
                # cur.execute(query_exp_del, (dataset.user_id, dataset.dataset_name)) 
                
                # 주어진 dataset 추가 
                query, values = await insert_from_dict(row=row_dict, table='Datasets') 
                await cur.execute(query, values)
                result = dataset
            
            else: 
                result = 'DATASET ALREADY EXISTS!'
        await conn.commit()

    # DATASETS[dataset.user_id] = dataset 

    return result # library 쪽으로 return (필요 없을수도)


async def upload_experiment(experiment: Experiment,
                            connection=Depends(get_db)) -> Experiment:
    
    ### TODO: experiment -> s3, 주소 변수로 저장
    ### TODO: 지표 계산 (dataset: Dataset 필요) 

    primary_values = (experiment.user_id, experiment.dataset_name, experiment.experiment_name, 
                                experiment.alpha, experiment.objective_fn)
    
    # experiment id (auto_increment) 값 얻기 위한 insert
    async with connection as conn:
        async with conn.cursor() as cur:
            init_query = 'INSERT INTO Experiments (user_id, dataset_name, experiment_name, alpha, objective_fn) \
                    VALUES (%s, %s, %s, %s, %s)'
            await cur.execute(init_query, primary_values) 

        await conn.commit()

    async with connection as conn:
        async with conn.cursor() as cur:
            id_query = 'SELECT experiment_id FROM Experiments WHERE \
                    user_id=%s, dataset_name=%s, experiment_name=%s, alpha=%s, objective_fn=%s'
            await cur.execute(id_query, primary_values)
            result = await cur.fetchone()

    # experiment_id로 s3에 저장
    exp_key = str(result['experiment_id'])
    row_dict = s3_transmission(experiment, exp_key)

    # 나머지 항목들 null 에서 s3 이름으로 업데이트
    placeholders = ', '.join('{}=%s'.format(k) for k in row_dict.keys())

    async with connection as conn:
        async with conn.curosr() as cur:
            s3_insert_query = 'UPDATE Experiment SET {}'.format(placeholders)
            await cur.execute(s3_insert_query, tuple(row_dict.values()))


async def get_metrics():
    pass

