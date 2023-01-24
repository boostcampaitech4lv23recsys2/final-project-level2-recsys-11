from asyncmy import connect

import logging
import sys


from fastapi import APIRouter

rds_config = {} # database 정보


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
