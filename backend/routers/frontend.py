<<<<<<< HEAD
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator, EmailStr
from starlette import status

router = APIRouter()

class UserCreate(BaseModel):
    username: str
    password1: str
    password2: str
    email: EmailStr

    @validator('username', 'password1', 'password2', 'email')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

    @validator('password2')
    def passwords_match(cls, v, values):
        if 'password1' in values and v != values['password1']:
            raise ValueError('비밀번호가 일치하지 않습니다')
        return v

@router.post("/create_user")
def user_create(_user_create: UserCreate):
    from dependencies import user_db
    username = _user_create.username

    if username in user_db.id.unique():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="이미 존재하는 사용자입니다.")
    password = _user_create.password1
    email = _user_create.email

    user_db = user_db.append({'id':username, 'password':password, 'email':email}, ignore_index=True)

    return user_db.to_dict('records')
=======
from asyncmy.cursors import DictCursor
from fastapi import APIRouter, Depends
from typing import Dict

from routers.database import get_db_dep

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


>>>>>>> 6eb592e918aa33aed8eacf0073eed9cab73c2910
