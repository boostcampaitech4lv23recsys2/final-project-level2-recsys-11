from asyncmy.cursors import DictCursor
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from starlette import status
from typing import Dict
from fastapi.responses import JSONResponse

from routers.database import get_db_inst, get_db_dep

router = APIRouter()  

#### PAGE 1
## Compare Table dataset 

@router.get('/get_exp_total')
async def get_exp_total(ID: str, dataset_name:str):
    total_exps = await get_total_info(ID, dataset_name) 

    if not total_exps:
         return {'msg': 'Experiments Not Found'}
    
    return total_exps # metric_per_user S3 object name included


#### PAGE 2
## Model vs. Model

# 페이지 2로 바로 넘어가자마자 실행 - exp_ids: List(pre-selected exp_ids from Page 1)
# [COMPARE!] 버튼 누르기 전 디스플레이 용
@router.get('/selected_models')
async def selected_models(ID:str, dataset_name:str, exp_ids: List[int] = Query(default=None)):
    total_exps = await get_total_info(ID, dataset_name) # cached

    if not total_exps:
         return {'msg': 'Experiments Not Found'}
    
    total_exps_pd = pd.DataFrame(total_exps).set_index('exp_id')
    selected_rows = total_exps_pd[['experiment_name', 'alpha', 'objective_fn', 'hyperparameters']].iloc[exp_ids]

    return selected_rows.to_dict('tight')

# COMPARE 누른 후  
@router.get('/selected_metrics')
async def selected_metrics(ID:str, dataset_name:str, exp_ids: List[int] = Query(default=None)):
    total_exps = await get_total_info(ID, dataset_name) # cached
    total_exps_pd = pd.DataFrame(total_exps).set_index('exp_id')

    model_metrics = total_exps_pd[['recall', 'map', 'ndcg', 'tail_percentage', 'avg_popularity', 'coverage',
                                  'diversity_cos', 'diversity_jac', 'serendipity_pmi', 'serendipity_jac', 
                                  'novelty']].iloc[exp_ids,:]

    user_metric_s3 = total_exps_pd.iloc[exp_ids]['metric_per_user'].to_dict()
    user_metrics = {str(model_id): await get_from_s3(s3_loc) for model_id, s3_loc in user_metric_s3.items()}

    result = {'model_metrics': model_metrics.to_dict(orient='tight'), 
              'user_metrics': user_metrics}

    return result


#### PAGE 4
##  Deep Analysis: User, Item

@router.get('/user_info')
async def user_info(ID: str, dataset_name: str, exp_id: int):
    # GET: dataset - user_side (age, gender, occupation)
    #              - train_df (->user_profile)
    #              
    # GET: exp(exp-id) - pred_items
    #                  - xs,ys (user)

    df_row = await get_df(ID, dataset_name) 
    
    if df_row == None:
        return {'msg': 'Dataset Not Found'}
    exp_row = await get_exp(exp_id) 
    if exp_row == None:
        return {'msg': 'Model Not Found'}

    user_side = await s3_to_pd(key_hash=df_row['user_side'])
    user_side_pd = user_side[['user_id', 'gender', 'age', 'occupation']]
    user_profile_pd = await inter_to_profile(key_hash=df_row['train_interaction'], group_by='user_id', col='item_id') 

    pred_item_pd = await s3_to_pd(key_hash=exp_row['pred_items'])
    user_tsne_pd = await s3_to_pd(key_hash=exp_row['user_tsne'])

    dfs = [user_side_pd, user_profile_pd, pred_item_pd, user_tsne_pd]
    user_merged = reduce(lambda left,right: pd.merge(left,right,on='user_id'), dfs)

    return user_merged
    

