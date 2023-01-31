from asyncmy.cursors import DictCursor
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from starlette import status
from schemas.user import UserCreate
from typing import Dict
from fastapi.responses import JSONResponse

from cruds.database import check_user, get_exp, get_df
from database.rds import get_db_dep

router = APIRouter()  


@router.post("/create_user", status_code=202)
async def create_user(_user_create: UserCreate, connection=Depends(get_db_dep)):
    print('1023ukvdfljvhoire;')
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
                     recall, map, ndcg, tail_per, avg_pop, coverage, \
                     diversity_cos, diversity_jac, serendipity_pmi, serendipity_jac, novelty \
                    FROM Experiments WHERE ID = %s AND dataset_name = %s'
            await cur.execute(query, (ID, dataset_name))
            result = cur.fetchall()

    return result 
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
    df_row = get_df(); exp_row = get_exp()
    


@router.get('/item_info')
async def item_info(ID: str, dataset_name: str, exp_id: int):
    # TODO 
    # GET: image_uri
    # GET: dataset - item_side (genre, title, year, popularity)
    #              - train_df (->item_profile)
    # GET: exp(exp_id)    - xs, ys (item)
    #                     - pred_items (item_recommended users)
    df_row = get_df(); exp_row = get_exp()
    

    pass