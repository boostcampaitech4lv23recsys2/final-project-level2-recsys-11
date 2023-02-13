from pydantic import BaseModel, validator, Field

from uuid import UUID, uuid4

class UserCreate(BaseModel):
    ID: str
    password1: str
    password2: str
    api_token: UUID = Field(default_factory=uuid4)

    @validator('ID', 'password1', 'password2')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

    @validator('password2')
    def passwords_match(cls, v, values):
        if 'password1' in values and v != values['password1']:
            raise ValueError('비밀번호가 일치하지 않습니다')
        return v

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str