from asyncmy.cursors import DictCursor
from typing import Dict
from routers.database import get_db_inst

# 유저가 등록되있는지 여부 확인
async def check_user(ID:str) -> Dict:
    conn2 = get_db_inst()
    
    async with conn2 as conn:
        async with conn.cursor(cursor=DictCursor) as cur:
            query = "SELECT ID FROM Users WHERE ID = %s"
            await cur.execute(query, (ID,))
            result = await cur.fetchone()
    return result 