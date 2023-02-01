from asyncmy.cursors import DictCursor
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from starlette import status
from typing import Dict
from fastapi.responses import JSONResponse

from routers.database import get_db_inst, get_db_dep

router = APIRouter()  
    

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


