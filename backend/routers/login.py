from typing import List, Dict
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations

from fastapi import APIRouter

from routers.data import dataset_info

from typing import Dict, TypeVar

router = APIRouter()