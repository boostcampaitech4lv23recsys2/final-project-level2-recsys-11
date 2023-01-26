from functools import lru_cache
from pydantic import BaseSettings

class Settings(BaseSettings):
    host: str
    port: int
    user: str
    db: str
    password: str

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


@lru_cache
def get_rds_settings():
    return Settings(_env_file='rds.env', _env_file_encoding='utf-8')

