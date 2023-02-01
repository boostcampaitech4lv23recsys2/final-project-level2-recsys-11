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
<<<<<<< HEAD
=======

class Login_Settings(BaseSettings):
        SALT: str
        ACCESS_TOKEN_EXPIRE_MINUTES: int
        SECRET_KEY: str
        ALGORITHM: str
        
        class config:
                env_flie = '.env'
                env_flie_encoding = 'utf-8'


@lru_cache
def get_rds_settings():
    return RDS_Settings(_env_file='rds.env', _env_file_encoding='utf-8')

@lru_cache
def get_login_settings():
    return Login_Settings(_env_file='login.env', _env_file_encoding='utf-8')


@lru_cache
def get_s3_settings():
    return S3_Settings(_env_file='s3.env', _env_file_encoding='utf-8')

>>>>>>> staging
