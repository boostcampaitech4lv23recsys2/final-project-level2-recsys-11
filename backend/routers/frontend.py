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