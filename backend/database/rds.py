from asyncmy import connect
import logging
import sys

from schemas.config import get_rds_settings


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
