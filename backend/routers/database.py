from asyncmy import connect
import logging
import sys

from fastapi import APIRouter
from schemas.config import get_rds_settings

rds_config = get_rds_settings().dict() # database 정보

router = APIRouter()

def get_db():
    db = connect(**rds_config)
    try:
        yield db

    except:
        logging.error("RDS Not Connected")
        sys.exit(1)

    finally:
        db.close()


def get_metrics