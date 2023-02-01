from asyncmy.cursors import DictCursor
from datetime import datetime
from fastapi import APIRouter, Depends, Query 
from starlette import status
from schemas.user import UserCreate
from typing import Dict, List
from fastapi.responses import JSONResponse
import pandas as pd
from functools import reduce

from cruds.database import check_user, get_exp, get_df, get_total_info, inter_to_profile
from cruds.metrics import avg_metric, predicted_per_item
from database.rds import get_db_dep
from database.s3 import get_from_s3, s3_dict_to_pd, s3_to_pd

router = APIRouter()  

#### LOGIN 
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


#### PAGE 1
## Compare Table dataset 

@router.get('/get_exp_total')
async def get_exp_total(ID: str, dataset_name:str) -> Dict:
    total_exps = await get_total_info(ID, dataset_name)
    return total_exps # metric_per_user S3 object name included


#### PAGE 2
## Model vs. Model

# 페이지 2로 바로 넘어가자마자 실행 - exp_ids: List(pre-selected exp_ids from Page 1)
@router.get('/selected_models')
async def selected_models(ID:str, dataset_name:str, exp_ids: List[int] = Query(default=None)):
    total_exps = await get_total_info(ID, dataset_name)
    total_exps_pd = pd.DataFrame(total_exps).set_index('exp_id')

    selected_rows = total_exps_pd.iloc[exp_ids]

    return selected_rows.to_dict('tight')


#### PAGE 4
##  Deep Analysis: User, Item

@router.get('/user_info')
async def user_info(ID: str, dataset_name: str, exp_id: int):
    # GET: dataset - user_side (age, gender, occupation)
    #              - train_df (->user_profile)
    #              
    # GET: exp(exp-id) - pred_items
    #                  - xs,ys (user)
    #
    df_row = await get_df(ID, dataset_name)
    exp_row = await get_exp(exp_id)

    user_side_pd = await s3_to_pd(key_hash=df_row['user_side'])[['user_id', 'gender', 'age', 'occupation']]
    user_profile_pd = await inter_to_profile(key_hash=df_row['train_interaction'], group_by='user_id', col='item_id') 

    pred_item_pd = await s3_to_pd(key_hash=exp_row['pred_items'])
    user_tsne_pd = await s3_to_pd(key_hash=exp_row['user_tsne'])

    dfs = [user_side_pd, user_profile_pd, pred_item_pd, user_tsne_pd]
    user_merged = reduce(lambda left,right: pd.merge(left,right,on='user_id'), dfs)

    return user_merged
    

@router.get('/item_info')
async def item_info(ID: str, dataset_name: str, exp_id: int):
    # GET: image_uri
    # GET: dataset - item_side (genre, title, year, popularity)
    #              - train_df (->item_profile)
    # GET: exp(exp_id)    - pred_items (->item_recommended users)
    #                     - xs, ys (item)
    #                     
    df_row = await get_df(ID, dataset_name)
    exp_row = await get_exp(exp_id)
    
    item_side_pd = await s3_to_pd(key_hash=df_row['item_side'])[['genre', 'title', 'year', 'popularity']]
    item_profile_pd = await inter_to_profile(key_hash=df_row['train_interaction'], group_by='item_id', col='user_id') 

    item_rec_users_pd = await predicted_per_item(pred_item_hash=exp_row['pred_items'])
    item_tsne_pd = await s3_to_pd(key_hash=exp_row['item_tsne'])

    dfs = [item_side_pd, item_profile_pd, item_rec_users_pd, item_tsne_pd]
    item_merged = reduce(lambda left,right: pd.merge(left,right,on='item_id'), dfs)

    return item_merged
