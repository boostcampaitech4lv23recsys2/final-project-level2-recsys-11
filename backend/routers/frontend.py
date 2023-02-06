from asyncmy.cursors import DictCursor
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
import json
from starlette import status
from schemas.user import UserCreate
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
from functools import reduce

from cruds.database import (
    get_exp,
    get_df,
    get_total_info,
    inter_to_profile,
    get_total_reranked,
)
from cruds.metrics import predicted_per_item, recall_per_user, get_metric_per_users
from database.rds import get_db_dep
from database.s3 import get_from_s3, s3_dict_to_pd, s3_to_pd

from engine.distance import get_distance_mat, get_jaccard_mat
from engine.metric import get_total_information
from engine.rerank import get_total_reranks

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
            query = "SELECT dataset_name FROM Datasets WHERE ID = %s"
            await cur.execute(query, (ID,))
            result = await cur.fetchall()

    if result:
        return [row["dataset_name"] for row in result]
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
            "novelty",
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
async def reranked_exp(
    ID: str, dataset_name: str, exp_names: Tuple[str] = Query(default=None)
):
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
            "novelty",
        ]
    ]

    user_metrics = await get_metric_per_users(reranked_pd)

    metrics = {
        "model_info": model_info.to_dict(orient="tight"),
        "user_metrics": user_metrics.to_dict(orient="tight"),
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
    pred_item_pd["pred_items"] = pred_item_pd["pred_items"].apply(
        lambda x: x[:10]
    )  ######

    user_tsne_pd = await s3_to_pd(key_hash=exp_row["user_tsne"])
    user_tsne_pd = user_tsne_pd.rename(columns={"item_id": "user_id"})
    # pred_item_pd = pred_item_pd.rename_axis('user_id').reset_index()
    # user_tsne_pd = user_tsne_pd.rename_axis('user_id').reset_index()

    recall_per_user_pd = await recall_per_user(key_hash=exp_row["metric_per_user"])

    dfs = [
        user_side_pd,
        user_profile_pd,
        pred_item_pd,
        user_tsne_pd,
        recall_per_user_pd,
    ]
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
    if df_row == None:
        return {"msg": "Dataset Not Found"}

    exp_row = await get_exp(exp_id)
    if exp_row == None:
        return {"msg": "Model Not Found"}

    item_side = await s3_to_pd(key_hash=df_row["item_side"])
    item_side_pd = item_side[
        ["item_id", "item_name", "genres:multi", "year", "item_popularity", 'item_url']
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
    item_merged = reduce(
        lambda left, right: pd.merge(left, right, on="item_id", how="outer"), dfs
    )
    item_merged = item_merged.fillna("")

    return item_merged.to_dict(orient="tight")


@router.get("/rerank_users")
async def rerank_users(
    ID: str,
    dataset_name: str,
    exp_id: int,
    n_candidates: int,
    objective_fn: str,
    alpha: float,
    user_ids: List[str] = Query(default=["1", "2", "3", "5"]),
):

    df_row = await get_df(ID, dataset_name)
    if df_row == None:
        return {"msg": "Dataset Not Found"}

    exp_row = await get_exp(exp_id)
    if exp_row == None:
        return {"msg": "Model Not Found"}

    base_pred_items = await s3_to_pd(key_hash=exp_row["pred_items"])
    base_pred_scores = await s3_to_pd(key_hash=exp_row["pred_scores"])

    need_items = list(
        set(np.array(base_pred_items["pred_items"].values.tolist()).flatten().tolist())
    )
    need_users = user_ids

    train_interaction = await s3_to_pd(key_hash=df_row["train_interaction"])

    prediction_matrix = pd.DataFrame(
        data=0.0,
        index=train_interaction["user_id"].unique(),
        columns=train_interaction["item_id"].unique(),
    )

    prediction_matrix_before = pd.merge(base_pred_items, base_pred_scores, on="user_id")

    for _, row in prediction_matrix_before.iterrows():
        if row["user_id"] in need_users:
            prediction_matrix.loc[row["user_id"], row["pred_items"]] = row[
                "pred_scores"
            ]

    train_interaction = await s3_to_pd(key_hash=df_row["train_interaction"])
    # encoding
    pred_users = prediction_matrix.index.tolist()
    pred_items = prediction_matrix.columns.tolist()  # 현재 사실 모든 아이템
    pred_mat = prediction_matrix.values

    uid2idx = {v: k for k, v in enumerate(pred_users)}
    iid2idx = {v: k for k, v in enumerate(pred_items)}

    idx2uid = {k: v for k, v in enumerate(pred_users)}
    idx2iid = {k: v for k, v in enumerate(pred_items)}

    train_interaction = train_interaction[
        train_interaction["user_id"].isin(pred_users)
    ]  # 이게 문제
    train_interaction = train_interaction[train_interaction["item_id"].isin(pred_items)]
    train_interaction["user_idx"] = train_interaction["user_id"].map(uid2idx)
    train_interaction["item_idx"] = train_interaction["item_id"].map(iid2idx)

    # train_interaction2 = train_interaction[train_interaction['item_id'].isin(need_items)]

    # distance matrix ready
    item_side = await s3_to_pd(key_hash=df_row["item_side"])
    cos_dist, pmi_dist = await get_distance_mat(train_interaction)
    jac_dist = None
    if "item_vector" in item_side.columns:
        item_vector = (
            item_side[["item_id", "item_vector"]].set_index("item_id").squeeze()
        )
        item_vector.index = item_vector.index.map(iid2idx)
        jac_dist = await get_jaccard_mat(item_vector)

    # quant prepare ready
    user_profile = train_interaction.groupby("user_idx")["item_idx"].apply(list)
    item_popularity = (
        train_interaction.groupby("item_idx")["user_idx"].count()
        / train_interaction["user_idx"].nunique()
    )
    tail_items = item_popularity.index[-int(len(item_popularity) * 0.8) :].tolist()
    total_items = train_interaction["item_idx"].unique()

    ground_truth = await s3_to_pd(key_hash=df_row["ground_truth"])
    ground_truth = ground_truth[ground_truth["user_id"].isin(pred_users)]
    ground_truth = ground_truth[ground_truth["item_id"].isin(pred_items)]

    ground_truth["user_idx"] = ground_truth["user_id"].map(uid2idx)
    ground_truth["item_idx"] = ground_truth["item_id"].map(iid2idx)

    actuals = ground_truth.groupby("user_idx")["item_idx"].apply(list)

    actuals = [
        items
        for user, items in actuals.iteritems()
        if user in [uid2idx[nu] for nu in need_users]
    ]

    # candidates = np.argsort(-pred_mat, axis=1)[[uid2idx[nu] for nu in need_users]][ :n_candidates]
    # print(actuals)

    candidates = np.array(
        [
            [iid2idx[item] for item in items]
            for i, (user_id, items) in base_pred_items.iterrows()
            if user_id in need_users
        ]
    )

    candidates = candidates[:, :n_candidates]

    # basic
    metrices, _ = await get_total_information(
        predicts=candidates,
        actuals=actuals,
        cos_dist=cos_dist,
        pmi_dist=pmi_dist,
        user_profile=user_profile,
        item_popularity=item_popularity,
        tail_items=tail_items,
        total_items=total_items,
        jac_dist=jac_dist,
        k=10,
    )

    if "cos" in objective_fn:
        dist_mat = cos_dist
    elif "pmi" in objective_fn:
        dist_mat = pmi_dist
    else:
        dist_mat = jac_dist

    objective_fn = objective_fn if objective_fn == "novelty" else objective_fn[:-5]
    rerank_predicts = await get_total_reranks(
        mode=objective_fn,
        candidates=candidates,
        prediction_mat=pred_mat,
        distance_mat=dist_mat,
        user_profile=user_profile,
        item_popularity=item_popularity,
        alpha=alpha,
        k=10,
    )

    rerank_metrices, _ = await get_total_information(
        predicts=rerank_predicts,
        actuals=actuals,
        cos_dist=cos_dist,
        pmi_dist=pmi_dist,
        user_profile=user_profile,
        item_popularity=item_popularity,
        tail_items=tail_items,
        total_items=total_items,
        jac_dist=jac_dist,
        k=10,
    )

    print(metrices)
    print(rerank_metrices)

    # rerank predicts decode
    decode_re_predicts = np.vectorize(lambda x: idx2iid[x])(rerank_predicts)
    decode_re_pred_items = pd.Series(
        {i: v.tolist() for i, v in enumerate(decode_re_predicts)}
    )
    decode_re_pred_items.index = need_users
    decode_re_pred_items_df = pd.DataFrame(decode_re_pred_items, columns=["pred_items"])

    decode_re_pred_items_df = decode_re_pred_items_df.reset_index()
    decode_re_pred_items_df = decode_re_pred_items_df.rename(
        columns={"index": "user_id"}
    )

    met = pd.concat([metrices, rerank_metrices], axis=1).T
    met.index = ["origin", "rerank"]

    return {
        "metric_diff": met.to_dict(orient="tight"),
        "rerank": decode_re_pred_items_df.to_dict(orient="tight"),
    }
    # 필요한 아이템들로만 distance mat 구성

    # uid2idx = {v: k for k, v in enumerate(pred_users)}
    # iid2idx = {v: k for k, v in enumerate(pred_items)}

    # idx2uid = {k: v for k, v in enumerate(pred_users)}
    # idx2iid = {k: v for k, v in enumerate(pred_items)}

    # train_interaction = train_interaction[train_interaction['user_id'].isin(pred_users)]
    # train_interaction = train_interaction[train_interaction['item_id'].isin(pred_items)]
    # train_interaction['user_idx'] = train_interaction['user_id'].map(uid2idx)
    # train_interaction['item_idx'] = train_interaction['item_id'].map(iid2idx)

    # if 'jac' in objective_fn:
    #     item_side = await s3_to_pd(key_hash=df_row['item_side'])
    #     item_vector = item_side[['user_id', 'item_vector']]
