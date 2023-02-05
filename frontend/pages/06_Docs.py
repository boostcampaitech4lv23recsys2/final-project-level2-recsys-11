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

dash.register_page(__name__, path='/Docs')

layout = dcc.Markdown(
    '''
    # FAQ
    
    ## 1. 정량지표와 정성지표란?

    ## 2. Reranking이란?

    ## 3. Web4Rec Library란?

    ## 4. Item vector와 t-sne는 어떻게 이루어지나요?

    '''
)