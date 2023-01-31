from asyncmy import connect
from functools import lru_cache
import logging
import sys

from schemas.config import RDS_Settings

@lru_cache(maxsize=1)
def get_rds_settings():
    return RDS_Settings(_env_file='rds.env', _env_file_encoding='utf-8')

rds_config = get_rds_settings().dict() # RDS database 정보


def get_db_inst():
    return connect(**rds_config)


def get_db_dep():
    db = connect(**rds_config)
    try:
        yield db

    except ValueError:
        logging.error("Validation Error: ValueError - DB Connection")
        return None

    except:
        logging.error("RDS Not Connected")
        sys.exit(1)

    finally:
        db.close()
