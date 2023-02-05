from asyncmy.cursors import DictCursor
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
import json
from starlette import status
from schemas.user import UserCreate
from typing import Dict, Tuple, List

import pandas as pd
from functools import reduce

from cruds.database import get_exp, get_df, get_total_info, inter_to_profile, get_total_reranked
from cruds.metrics import predicted_per_item, recall_per_user, get_metric_per_users
from database.rds import get_db_dep
from database.s3 import get_from_s3, s3_dict_to_pd, s3_to_pd

router = APIRouter(prefix="/frontend")

#### PAGE 1
## Compare Table dataset

@router.get("/get_exp_total", status_code=201)
async def get_exp_total(ID: str, dataset_name: str):
    total_exps = await get_total_info(ID, dataset_name)

    if not total_exps:
        return {"msg": "Experiments Not Found"}

    total_exps_pd = pd.DataFrame(total_exps)
    total_exps_pd.drop("metric_per_user", axis=1, inplace=True)

    return total_exps_pd.to_dict(orient="tight")

@router.get("/check_dataset", status_code=201)
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

#### PAGE 2
## Model vs. Model

# 페이지 2로 바로 넘어가자마자 실행 - exp_ids: List(pre-selected exp_ids from Page 1)
# [COMPARE!] 버튼 누르기 전 디스플레이 용
@router.get("/selected_models")
async def selected_models(
    ID: str, dataset_name: str, exp_ids: List[int] = Query(default=None)
):
    total_exps = await get_total_info(ID, dataset_name)  # cached

    if not total_exps:
        return {"msg": "Experiments Not Found"}

    total_exps_pd = pd.DataFrame(total_exps).set_index("exp_id")
    selected_rows = total_exps_pd[
        ["experiment_name", "alpha", "objective_fn", "hyperparameters"]
    ].loc[exp_ids]

    return selected_rows.to_dict("tight")


# COMPARE 누른 후
@router.get("/selected_metrics")
async def selected_metrics(
    ID: str, dataset_name: str, exp_ids: List[int] = Query(default=None)
):
    total_exps = await get_total_info(ID, dataset_name)  # cached
    total_exps_pd = pd.DataFrame(total_exps).set_index("exp_id").loc[exp_ids]

    model_metrics = total_exps_pd[
        [
            "recall",
            "map",
            "ndcg",
            "tail_percentage",
            "avg_popularity",
            "coverage",
            "diversity_cos",
            "diversity_jac",
            "serendipity_pmi",
            "serendipity_jac",
            "novelty"
        ]
    ]

    user_metrics = await get_metric_per_users(total_exps_pd)

    metrics = {
        "model_metrics": model_metrics.to_dict(orient="tight"),
        "user_metrics": user_metrics.to_dict(orient="tight"),
    }

    return metrics


#### PAGE 3
## Reranking

@router.get("/reranked_exp")
async def reranked_exp(ID: str, dataset_name: str, exp_names: Tuple[str] = Query(default=None)):
    exps = await get_total_reranked(ID, dataset_name, exp_names)
    reranked_pd = pd.DataFrame(exps)
   
    model_info = reranked_pd[
       [
            "experiment_name",
            "alpha",
            "objective_fn",
            "recall",
            "map",
            "ndcg",
            "tail_percentage",
            "avg_popularity",
            "coverage",
            "diversity_cos",
            "diversity_jac",
            "serendipity_pmi",
            "serendipity_jac",
            "novelty"
       ]
   ] 

    user_metrics = await get_metric_per_users(reranked_pd)

    metrics = {
       "model_info": model_info.to_dict(orient="tight"),
       "user_metrics": user_metrics.to_dict(orient="tight")
   }

    return metrics


#### PAGE 4
##  Deep Analysis: User, Item

@router.get("/user_info")
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
    user_tsne_pd = user_tsne_pd.rename(columns={'item_id':'user_id'})
    # pred_item_pd = pred_item_pd.rename_axis('user_id').reset_index()
    # user_tsne_pd = user_tsne_pd.rename_axis('user_id').reset_index()

    recall_per_user_pd = await recall_per_user(key_hash=exp_row['metric_per_user'])

    dfs = [user_side_pd, user_profile_pd, pred_item_pd, user_tsne_pd, recall_per_user_pd]
    user_merged = reduce(lambda left,right: pd.merge(left,right,on='user_id'), dfs)

    return user_merged.to_dict(orient='tight')


@router.get("/item_info")
async def item_info(ID: str, dataset_name: str, exp_id: int):
    # GET: image_uri
    # GET: dataset - item_side (genre, title, year, popularity)
    #              - train_df (->item_profile)
    # GET: exp(exp_id)    - pred_items (->item_recommended users)
    #                     - xs, ys (item)

    df_row = await get_df(ID, dataset_name)
    if df_row == None:
        return {"msg": "Dataset Not Found"}

    exp_row = await get_exp(exp_id)
    if exp_row == None:
        return {"msg": "Model Not Found"}

    item_side = await s3_to_pd(key_hash=df_row["item_side"])
    item_side_pd = item_side[
        ["item_id", "item_name", "genres:multi", "year", "item_popularity"]
    ]
    item_profile_pd = await inter_to_profile(
        key_hash=df_row["train_interaction"], group_by="item_id", col="user_id"
    )
    item_profile_pd.columns = ["item_id", "item_profile"]

    item_rec_users_pd = await predicted_per_item(pred_item_hash=exp_row["pred_items"])
    item_tsne_pd = await s3_to_pd(key_hash=exp_row["item_tsne"])
    # item_tsne_pd = item_tsne_pd.rename_axis('item_id').reset_index()

    dfs = [item_side_pd, item_profile_pd, item_rec_users_pd, item_tsne_pd]
    # print([i.describe() for i in dfs])
    item_merged = reduce(lambda left, right: pd.merge(left, right, on="item_id"), dfs)

    return item_merged.to_dict(orient="tight")

