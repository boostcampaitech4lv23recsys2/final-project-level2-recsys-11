from asyncmy.cursors import DictCursor
from datetime import datetime
from fastapi import APIRouter, Depends
from starlette import status
from schemas.user import UserCreate
from typing import Dict
from fastapi.responses import JSONResponse
import pandas as pd

from cruds.database import check_user, get_exp, get_df, inter_to_profile
from cruds.metrics import avg_metric
from database.rds import get_db_dep
from database.s3 import get_from_s3, s3_dict_to_pd

router = APIRouter()  


@router.post("/create_user", status_code=202)
async def create_user(_user_create: UserCreate, connection=Depends(get_db_dep)):

    # connection = get_db_inst()
    user = await check_user(_user_create.ID) 

    if user:
        return JSONResponse({'msg':'error'},status_code=status.HTTP_409_CONFLICT)

    else:
        async with connection as conn:
            async with conn.cursor() as cur:
                curr_time = datetime.now()
                query = "INSERT INTO Users (ID, password, access_time) VALUES (%s, %s, %s)"
                await cur.execute(query, (_user_create.ID, _user_create.password1, curr_time))
            await conn.commit()

        return {'message': f"User: {_user_create.ID} has been ADDED"}
    

@router.get('/get_exp_total')
async def get_exp_total(ID: str, dataset_name:str, connection=Depends(get_db_dep)) -> Dict:
    async with connection as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = 'SELECT experiment_name, alpha, objective_fn, hyperparameters, \
                     recall, ap, ndcg, tail_percentage, avg_popularity, coverage, \
                     diversity_cos, diversity_jac, serendipity_pmi, serendipity_jac, novelty \
                    FROM Experiments WHERE ID = %s AND dataset_name = %s'
            await cur.execute(query, (ID, dataset_name))
            result = cur.fetchall()

    rows_with_avg = avg_metric(result)

    return rows_with_avg 
    # [{'experiment_name': (name), 'alpha' : (alpha)}, {'experiment_name2': (name2), 'alpha2' : (alpha2)}, ...]


@router.get('/user_info')
async def user_info(ID: str, dataset_name: str, exp_id: int):
    # TODO
    # GET: dataset - user_side (age, gender, occupation)
    #              - train_df (->user_profile)
    #              
    # GET: exp(exp-id) - pred_items
    #                  - xs,ys (user)
    #
    df_row = await get_df(ID, dataset_name)
    exp_row = await get_exp(exp_id)

    user_side = await get_from_s3(key_hash=df_row['user_side'])
    user_side_pd = s3_dict_to_pd(user_side)['user_id', 'gender', 'age', 'occupation']

    user_profile = await inter_to_profile(key_hash=df_row['train_interaction'], group_by='user_id', col='item_id')              # pd.DataFrame

    m = pd.merge(user_side_pd, user_profile, on='user_id')


    pred_items = await get_from_s3(key_hash=exp_row['pred_items'])
    user_tsne = await get_from_s3(key_hash=exp_row['user_tsne'])

    return
    

@router.get('/item_info')
async def item_info(ID: str, dataset_name: str, exp_id: int):
    # TODO 
    # GET: image_uri
    # GET: dataset - item_side (genre, title, year, popularity)
    #              - train_df (->item_profile)
    # GET: exp(exp_id)    - pred_items (item_recommended users)
    #                     - xs, ys (item)
    #                     
    df_row = await get_df(ID, dataset_name)
    exp_row = await get_exp(exp_id)
    
    get_from_s3(key_hash=df_row['item_side'])
    item_profile = await inter_to_profile(key_hash=df_row['train_interaction'], group_by='item_id', col='user_id') # Dict

    get_from_s3(key_hash=exp_row['pred_items'])
    get_from_s3(key_hash=exp_row['item_tsne'])

