from functools import lru_cache
from pydantic import BaseSettings

class RDS_Settings(BaseSettings):
    host: str
    port: int
    user: str
    db: str
    password: str

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


class S3_Settings(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    region_name: str

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
