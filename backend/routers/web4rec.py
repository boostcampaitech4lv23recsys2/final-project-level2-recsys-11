from asyncmy.cursors import DictCursor
from datetime import datetime
from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from typing import List, Dict

from cruds.database import check_user, insert_from_dict
from database.rds import get_db_dep
from database.s3 import get_from_s3, s3_transmission
from schemas.data import Dataset, CoreDataset, Experiment


router = APIRouter()

# DATASETS = {} # 유저별 데이터셋 in-memory에 따로 저장`    `


# 유저 아이디랑 비번이 쿼리에..?? 이게맞나
@router.get("/login")
async def login(ID:str, password:str, connection=Depends(get_db_dep)) -> List:
    user = await check_user(ID) 
 
    # 데이터에서 유저 확인
    if user:
        async with connection as conn:
            async with conn.cursor() as cur:
                curr_time = datetime.now()
                query = "UPDATE Users SET access_time=%s WHERE ID=%s and password=%s"
                await cur.execute(query, (curr_time, ID, password))
            await conn.commit()
        
        return list(user.values()) # [{'ID': (ID)}]
    
    else:
        # content = {'message': f"Username: {ID} Not Found"}
        # return JSONResponse(content, status_code=404)
        return ['unknown']
    

@router.post("/add_user", status_code=202)
async def add_user(ID: str, password:str, connection=Depends(get_db_dep)): 
    async with connection as conn:
        async with conn.cursor() as cur:
            curr_time = datetime.now()
            query = "INSERT INTO Users (ID, password, access_time) VALUES (%s, %s, %s)"
            await cur.execute(query, (ID, password, curr_time))
        await conn.commit()
    
    return {'message': f"User: {ID} has been ADDED"}


@router.get("/check_dataset")
async def check_dataset(ID: str, connection=Depends(get_db_dep)) -> List:
    async with connection as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = 'SELECT dataset_name FROM Datasets WHERE ID = %s'
            await cur.execute(query, (ID,))
            result = await cur.fetchall() 

    if result:
        return [row['dataset_name'] for row in result]
    else:
        return []


# password도 필요하게 해주는게 좋을까
@router.delete('/delete_dataset')
async def delete_dataset(ID:str, dataset_name: str, connection=Depends(get_db_dep)):
    async with connection as conn:
        async with conn.cursor() as cur:
            query_dataset_del = 'DELETE FROM Datasets WHERE ID = %s and dataset_name = %s'
            query_exp_del = 'DELETE FROM Experiments WHERE ID = %s and dataset_name = %s'
            await cur.execute(query_dataset_del, (ID, dataset_name))
            await cur.execute(query_exp_del, (ID, dataset_name)) 
        await conn.commit() 

    return {'message': f"Dataset:{dataset_name} (User: {ID}) has been DELETED"}


@router.post("/upload_dataset", status_code=202)
async def upload_dataset(dataset: Dataset,
                         connection = Depends(get_db_dep)) -> Dict:

    primary_key = dataset.ID + '_' + dataset.dataset_name # str 
    row_dict = await s3_transmission(dataset, primary_key)
    row_dict['ID'] = dataset.ID
    row_dict['dataset_name'] = dataset.dataset_name
    row_dict['upload_time'] = dataset.upload_time
    row_dict['dataset_desc'] = dataset.dataset_desc

    query, values = await insert_from_dict(row=row_dict, table='Datasets') 

    async with connection as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, values)
        await conn.commit()

    # DATASETS[dataset.ID] = dataset 
 
    return row_dict # library 쪽으로 return (필요 없을수도)


@router.get('/download_core_dataset', status_code=202)
async def download_core_dataset(
    ID: str,
    dataset_name: str,
    connection=Depends(get_db_dep)
):
    """
    익스퍼리먼트는 train_interaction + ground_truth 가 필요하다.
    이들을 데이터셋 이름을 받고 던져준다. 로직상 이렇게 하겟다 알겠나? 단결
    """
    async with connection as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = "SELECT train_interaction, ground_truth, item_side FROM Datasets WHERE ID = %s AND dataset_name = %s"
            await cur.execute(query, (ID, dataset_name))
            result = await cur.fetchone()
            
    train_interaction_hash = result['train_interaction']
    train_interaction = await get_from_s3(train_interaction_hash)

    ground_truth_hash = result['ground_truth']
    ground_truth = await get_from_s3(ground_truth_hash)

    item_side_hash = result['item_side']
    item_side = await get_from_s3(item_side_hash)

    ret = {
        'train_interaction': train_interaction,
        'ground_truth': ground_truth,
        'item_side': item_side
    }
    return ret


@router.post("/upload_experiment", status_code=202)
async def upload_experiment(experiment: Experiment,
                            connection=Depends(get_db_dep)): #?
    
    primary_keys = ('ID', 'dataset_name', 'experiment_name', 'alpha', 'objective_fn')
    primary_values = (experiment.ID, experiment.dataset_name, experiment.experiment_name, 
                                str(experiment.alpha), str(experiment.objective_fn)) # 개별 experimentd의 고유 string 값들
    
    row_dict = await s3_transmission(experiment, "_".join(primary_values))
    # row_dict = dict(s3_dict) # dict of experiment
    row_dict.update({attribute: value for attribute, value in zip(primary_keys, primary_values)})

    row_dict['alpha'] = experiment.alpha
    row_dict['objective_fn'] = experiment.objective_fn

    row_dict['hyperparameters'] = experiment.hyperparameters

    row_dict['recall'] = experiment.recall
    row_dict['ndcg'] = experiment.ndcg
    row_dict['map'] = experiment.map
    row_dict['avg_popularity'] = experiment.avg_popularity
    row_dict['tail_percentage'] = experiment.tail_percentage
    row_dict['coverage'] = experiment.coverage

    row_dict['diversity_cos'] = experiment.diversity_cos
    row_dict['serendipity_pmi'] = experiment.serendipity_pmi
    row_dict['novelty'] = experiment.novelty

    row_dict['diversity_jac'] = experiment.diversity_jac
    row_dict['serendipity_jac'] = experiment.serendipity_jac
    
    query, values = await insert_from_dict(row=row_dict, table='Experiments')

    async with connection as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, values)
        await conn.commit()


    return row_dict