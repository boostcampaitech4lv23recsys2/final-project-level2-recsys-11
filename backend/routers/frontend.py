from asyncmy.cursors import DictCursor
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
import json
from starlette import status
from schemas.user import UserCreate
from typing import Dict, List

import pandas as pd
from functools import reduce

from cruds.database import check_user, get_exp, get_df, get_total_info, inter_to_profile
from cruds.metrics import avg_metric, predicted_per_item
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
    total_exps_pd = pd.DataFrame(total_exps).set_index("exp_id")

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
            "novelty",
        ]
    ].loc[exp_ids]

    index_id = list(user_metric_s3.keys())

    models = [await s3_to_pd(s3_loc) for s3_loc in user_metric_s3.values()]
    user_metrics = pd.concat(models, axis=1).T
    user_metrics.index = index_id

    result = {
        "model_metrics": model_metrics.to_dict(orient="tight"),
        "user_metrics": user_metrics.to_dict(orient="tight"),
    }

    return result


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
        return {"msg": "Dataset Not Found"}
    exp_row = await get_exp(exp_id)
    if exp_row == None:
        return {"msg": "Model Not Found"}

    user_side = await s3_to_pd(key_hash=df_row["user_side"])
    user_side_pd = user_side[["user_id", "gender", "age", "occupation"]]
    user_profile_pd = await inter_to_profile(
        key_hash=df_row["train_interaction"], group_by="user_id", col="item_id"
    )

    pred_item_pd = await s3_to_pd(key_hash=exp_row["pred_items"])
    user_tsne_pd = await s3_to_pd(key_hash=exp_row["user_tsne"])
    pred_item_pd = pred_item_pd.reset_index()
    user_tsne_pd = user_tsne_pd.reset_index()
    pred_item_pd.columns = ["user_id", "pred_items"]
    user_tsne_pd.columns = ["user_id", "xs", "ys"]

    dfs = [user_side_pd, user_profile_pd, pred_item_pd, user_tsne_pd]
    user_merged = reduce(lambda left, right: pd.merge(left, right, on="user_id"), dfs)

    return user_merged.to_dict(orient="tight")


@router.get("/item_info")
async def item_info(ID: str, dataset_name: str, exp_id: int):
    # GET: image_uri
    # GET: dataset - item_side (genre, title, year, popularity)
    #              - train_df (->item_profile)
    # GET: exp(exp_id)    - pred_items (->item_recommended users)
    #                     - xs, ys (item)

    df_row = await get_df(ID, dataset_name)
    exp_row = await get_exp(exp_id)

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
    item_tsne_pd = item_tsne_pd.reset_index()  # 얘도 나중에 수정
    item_tsne_pd.columns = ["item_id", "xs", "ys"]

    dfs = [item_side_pd, item_profile_pd, item_rec_users_pd, item_tsne_pd]
    item_merged = reduce(lambda left, right: pd.merge(left, right, on="item_id"), dfs)
    return item_merged.to_dict(orient="tight")
