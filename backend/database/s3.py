import boto3
import hashlib
import json
from typing import Dict, Union 

from schemas.data import Dataset, Experiment
from schemas.config import get_s3_settings

s3_config = get_s3_settings().dict() # S3 정보

client = boto3.client('s3', **s3_config)

async def send_to_s3(data: Dict, key_name: str) -> str:
    json_body = json.dumps(data)
    hash_object = hashlib.sha256(key_name.encode('utf-8'))
    key_hash = hash_object.hexdigest() + '.json'

    if len(key_hash) > 100:
        raise ValueError('Hash String Length Has Exceeded 100')

    client.put_object(Body=json_body, Bucket='mkdir-bucket', Key=key_hash)

    return key_hash 


async def get_from_s3(key_hash: str) -> Dict:   # key_hash = key_name.encode('utf-8') + json
    obj = client.get_object(Bucket='mkdir-bucket', Key=key_hash) 
    return json.loads(obj['Body'].read().decode('utf-8'))


async def s3_transmission(cls: Union[Dataset, Experiment], primary_key: str) -> Dict:
    row_dict = {} # attribute: key_hash (s3 file name)

    # json (library) -> json (s3) 
    for attribute, value in vars(cls).items():
        if isinstance(value, dict): 
            key_hash = await send_to_s3(data = value, key_name = primary_key+attribute)
            row_dict[attribute] = str(key_hash)
    
    return row_dict
