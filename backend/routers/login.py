from asyncmy.cursors import DictCursor
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from starlette import status
from schemas.user import UserCreate, Token
from typing import Dict
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
#pip install "python-jose[cryptography]"

from cruds.database import check_user, check_password
from routers.database import get_db_inst, get_db_dep

router = APIRouter()  

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
SECRET_KEY = "90854f928da3abee27b2a4af37b6b538308fdfe92538cfd0b8cf68a4c597cb4a"
ALGORITHM = "HS256"

@router.post("/create_user",)
async def create_user(_user_create: UserCreate, connection=Depends(get_db_dep)):

    # 이미 있는 아이디인지 확인
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

@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(),
                           ):

    # check user and password
    print(form_data.password)
    user = await check_password(form_data.username, form_data.password)
    print(user)
    # if not user or not (form_data.password, user.password):
    if not user:
        return JSONResponse({'msg': 'Invalid username or password'},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    # make access token
    data = {
        "sub": user['ID'],
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user['ID']
    }