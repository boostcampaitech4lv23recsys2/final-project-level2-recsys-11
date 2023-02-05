import dash
import json
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import dash_bootstrap_components as dbc

from dash import html, dcc, callback, Input, Output, State,  MATCH, ALL
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
from . import global_component as gct

dash.register_page(__name__, path='/FAQ')

layout = dcc.Markdown(
    '''
    # FAQ

    ## 1. 정량지표와 정성지표란?

    ## 2. Reranking이란?

    ## 3. Web4Rec Library란?

    ## 4. Item vector와 t-sne는 어떻게 이루어지나요?
    - prediction matrix 만들기
        - 모든 모델에서 Item vector를 추출하기 위해서는 prediction matrix가 필요합니다. 
        - prediction matrix란, 모델이 모든 User별로 예측한 Item의 선호도 혹은 score를 의미합니다. 
        - 또한, 생성된 prediction matrix는 dense한 형태이지만 threshold 값을 지정해 sparse matrix로 변경하여 메모리를 절약할 수도 있습니다.
    
    
    - embedding vector 생성하기
        - 저희 library에서는 2단계로 나누어 최종적으로 2차원 embedding vector를 생성합니다. 
        - 먼저 (user size, item size) 크기인 prediction matrix를 transpose를 취해 (item size, user size)로 변경 시켜준뒤, scikit-learn library의 pca를 활용해 50차원으로 축소시킵니다.
        - 그 다음으로, 50차원으로 축소된 embedding vector를 sklearn.manifold library를 활용해 2차원으로 축소합니다.
        - 이 때, 1차적으로 pca를 활용해 축소시키고 싶은 차원을 정할 수 있습니다
    '''
)