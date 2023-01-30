from asyncmy import connect
import boto3
import hashlib
import json
import logging
import sys
from typing import Any, Dict, List, Union, Tuple

JSON = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


from fastapi import APIRouter
from schemas.data import Dataset, Experiment
from schemas.config import get_rds_settings, get_s3_settings

rds_config = get_rds_settings().dict() # RDS database 정보
s3_config = get_s3_settings().dict() # S3 정보

client = boto3.client('s3', **s3_config)

router = APIRouter()


def get_db_inst():
    return connect(**rds_config)


def get_db_dep():
    db = connect(**rds_config)
    try:
        yield db

    except:
        logging.error("RDS Not Connected")
        sys.exit(1)

    finally:
        db.close()


async def send_to_s3(data: Dict, key_name: str) -> str:
    json_body = json.dumps(data)
    hash_object = hashlib.sha256(key_name.encode('utf-8'))
    key_hash = hash_object.hexdigest() + '.json'

    if len(key_hash) > 100:
        raise ValueError('Hash String Length Has Exceeded 100')

    client.put_object(Body=json_body, Bucket='mkdir-bucket', Key=key_hash)

    return key_hash 


async def get_from_s3(key_hash: str) -> Dict:   # key_hash = key_name.encode('utf-8')
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


async def insert_from_dict(row: Dict, table: str) -> Tuple:
    placeholders = ', '.join(['%s'] * len(row))
    columns = ', '.join(row.keys())
    query = "INSERT INTO %s ( %s ) VALUES ( %s )" % (table, columns, placeholders)

    return query, tuple(row.values())




    