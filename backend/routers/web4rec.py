from datetime import datetime

from fastapi import APIRouter, Depends

from routers.database import get_db

router = APIRouter()


# 유저가 등록되있는지 여부 확인

async def check_user(user_id, password, connection):
    async with connection as conn:
        async with conn.cursor() as cur:
            query = "SELECT EXISTS(SELECT 1 FROM User WHERE user_id = %s AND user_pwd = %s)"
            await cur.execute(query, (user_id, password))
            result = await cur.fetchone()
            return result


@router.post("/add_user")
async def add_user(user_id, password, connection=Depends(get_db)):
    async with connection as conn:
        async with conn.cursor() as cur:
            curr_time = await datetime.now()
            query = "INSERT INTO User (user_id, user_pwd, access_time) VALUES (%s, %s, %s)"
            await cur.execute(query, (user_id, password, curr_time))
            await conn.commit()


@router.get("/verify_user")
async def verify_user(user_id, password, connection=Depends(get_db)):
    if await check_user(user_id, password, connection):
        print(f'User Verified: {user_id}')
        return {'user': 'verified'}

        ## TODO
        ## pass user_id as CURRENT USER to other functions

    else: 
        print(f'User({user_id}) Not Found') 
        return {'user': 'not found'}