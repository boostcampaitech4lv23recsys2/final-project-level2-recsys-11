from asyncmy.cursors import DictCursor
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from starlette import status
from schemas.user import UserCreate, Token
from schemas.config import get_login_settings
from typing import Dict
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt, JWTError
#pip install "python-jose[cryptography]"

from cruds.database import check_user, check_password
from database.rds import get_db_inst, get_db_dep

router = APIRouter(prefix='/user')  

login_config = get_login_settings().dict()
ACCESS_TOKEN_EXPIRE_MINUTES = login_config['ACCESS_TOKEN_EXPIRE_MINUTES']
SECRET_KEY = login_config['SECRET_KEY']
ALGORITHM = login_config['ALGORITHM']

@router.post("/user_info")
async def create_user(_user_create: UserCreate, connection=Depends(get_db_dep)):

    # 이미 있는 아이디인지 확인
    user = await check_user(_user_create.ID) 

    if user:
        return JSONResponse({'msg':'error'},status_code=status.HTTP_409_CONFLICT)
        
    else:
        async with connection as conn:
            async with conn.cursor() as cur:
                curr_time = datetime.now()
                query = "INSERT INTO Users (ID, password, access_time, token) VALUES (%s, %s, %s, %s)"
                await cur.execute(query, (_user_create.ID, _user_create.password1, curr_time, _user_create.api_token))
            await conn.commit()

        return _user_create
        # return {'message': f"User: {_user_create.ID} has been ADDED"}


@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(),
                           ):
    # check user and password
    user = await check_password(form_data.username, form_data.password)
    
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

@router.post('/get_current_user', status_code=201)
async def get_current_user(token:Token, connection=Depends(get_db_dep)):
    print('token: ',token)
    credentials_exception = JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={'msg': 'Invalid username or password'},
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token.access_token, SECRET_KEY, algorithms=[ALGORITHM])
        print(payload)
        username: str = payload.get("sub")
        if username is None:
            return credentials_exception
    except JWTError:
        return credentials_exception
    
    async with connection as conn:
        async with conn.cursor() as cur:
            query = "SELECT ID FROM Users WHERE (ID) = %s"
            await cur.execute(query, (token.username))
            user = await cur.fetchone()

    if user is None:
        return credentials_exception
    