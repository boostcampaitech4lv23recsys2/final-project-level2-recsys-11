from asyncmy.cursors import DictCursor
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from starlette import status
from schemas.user import UserCreate
from typing import Dict

from cruds.database import check_user
from routers.database import get_db_dep


router = APIRouter()  


@router.post("/create_user", status_code=202)
async def create_user(_user_create: UserCreate, connection=Depends(get_db_dep)):
    user = await check_user(UserCreate.ID) 

    if user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="이미 존재하는 사용자입니다.")
    else:
        async with connection as conn:
            async with conn.cursor() as cur:
                curr_time = datetime.now()
                query = "INSERT INTO Users (ID, password, access_time) VALUES (%s, %s, %s)"
                await cur.execute(query, (_user_create.ID, _user_create.password1, curr_time))
            await conn.commit()

        return {'message': f"User: {_user_create.ID} has been ADDED"}
    

@router.get('/get_exp')
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


