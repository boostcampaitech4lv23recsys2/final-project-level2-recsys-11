import boto3
from functools import lru_cache
from async_lru import alru_cache
import hashlib
import json
from typing import Dict, Union 
import pandas as pd

from schemas.data import Dataset, Experiment
from schemas.config import S3_Settings


@lru_cache(maxsize=1)
def get_s3_settings():
    return S3_Settings(_env_file='s3.env', _env_file_encoding='utf-8')

s3_config = get_s3_settings().dict() # S3 정보
client = boto3.client('s3', **s3_config)


async def send_to_s3(data: Dict, key_name: str) -> str:
    json_body = json.dumps(data)
    hash_object = hashlib.sha256(key_name.encode('utf-8'))
    # key_hash = hash_object.hexdigest() + '.json'
    key_hash = key_name
    if len(key_hash) > 100:
        raise ValueError('Hash String Length Has Exceeded 100')

    client.put_object(Body=json_body, Bucket='mkdir-bucket-11', Key=key_hash)

    return key_hash 


@alru_cache(maxsize=10)
async def get_from_s3(key_hash: str) -> Dict:   # key_hash = key_name.encode('utf-8') + json
    obj = client.get_object(Bucket='mkdir-bucket-11', Key=key_hash) 
    return json.loads(obj['Body'].read().decode('utf-8'))


async def s3_dict_to_pd(s3_dict: Dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(s3_dict, orient='tight') 


async def s3_to_pd(key_hash:str) -> pd.DataFrame: 
    s3_obj = await get_from_s3(key_hash)
    s3_obj_pd = await s3_dict_to_pd(s3_obj)
    return s3_obj_pd


async def s3_transmission(cls: Union[Dataset, Experiment], primary_key: str) -> Dict:
    row_dict = {} # attribute: key_hash (s3 file name)

    # json (library) -> json (s3) 
    for attribute, value in vars(cls).items():
        if isinstance(value, dict): 
            key_hash = await send_to_s3(data = value, key_name = primary_key + '_' + attribute)
            row_dict[attribute] = str(key_hash)
    
    return row_dict