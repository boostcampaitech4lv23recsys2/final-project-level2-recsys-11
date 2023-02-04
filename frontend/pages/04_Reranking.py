import dash
import json
import copy
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
API_url = 'http://127.0.0.1:30004'
dash.register_page(__name__, path='/reranking')

total_metrics = None
total_metrics_users = None


metric_list = [
    'Diversity(jaccard)',
    'Diversity(cosine)',
    'Serendipity(jaccard)',
    'Serendipity(PMI)',
    'Novelty',
]



### alpha 선택하는 부분, radio
alpha_radio = html.Div([
    dbc.RadioItems(
            id="alpha",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "0.5", "value": 0.5},
                {"label": "1", "value": 1},  # 사용자가 지정한 alpha로 지정
            ],
            value=0.5,
                        ),
], className="radio-group")

model_form = html.Div([
    html.H6("Select Experiment"),
    dcc.Dropdown(id="selected_model_by_name"),
            html.Hr(),
            html.H6("Alpha: "),
            # html.P('Alpha:', className="p-0 m-0"),
            alpha_radio,
               html.H6("Select objective function(distance function)"),
    dcc.Checklist(
    metric_list,
    metric_list,
    id="obj_funcs",)
], className='form-style')

sidebar = html.Div(
    [
        html.H3("Select Options",),
        html.Hr(),
        html.Div(id='rerank_form', children=model_form),
        dbc.Button('Rerank!', id="rerank_btn", n_clicks=0, className="mt-3")
    ],
    className='sidebar'
)

total_graph = html.Div([
    html.Br(),
    html.H1(children='Reranking', style={'text-align': 'center','font-weight': 'bold'}),
    html.Hr(),
    
    html.Div(id='reranked_graph'),
    
    html.H3('Total Metric'),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Br(),
            ]),
            html.Div(id='total_metric')
            ]),
        ])
     ])

def specific_metric():
    specific_metric = html.Div([
        html.H3('Specific Metric'),
        dbc.Row([
            dbc.Col([
                dbc.RadioItems(
                    id="sort_of_metric",
                    className="btn-group",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-primary",
                    labelCheckedClassName="active",
                    options=[
                        {"label": "Qualitative", "value": 'Qual'},
                        {"label": "Quantitative", "value": 'Quant'},
                    ],
                    value='Qual',
                ),
                html.Br(),
                dcc.Dropdown(id='metric_list')
                ], width=3),
                html.Br(),
                html.Div([html.P(id="print_metric"),]),
            dbc.Col([
                dcc.Graph(id = 'bar_fig'), #figure=fig_qual
                html.Div(id = 'dist_fig'),  # dcc.Graph(id = 'dist_fig')
            ], width=8)
        ]),

        ],
        className="radio-group",
    )
    return specific_metric


layout = html.Div(children=[
    gct.get_navbar(has_sidebar=False),
    html.Div([
    sidebar,
    total_graph,
    html.Div(id = 'specific_metric_children')
    ]),
    html.Div(id='trash'),
    dcc.Store(id='store_selected_exp', storage_type='session'),
    dcc.Store(id='store_exp_names', storage_type="session"),
    dcc.Store(id='store_exp_ids', storage_type='session'),
    dcc.Store(id='store_selected_exp_names', data=[], storage_type='session')

], className="content")

### exp_names가 들어오면 실험 정보들 return 하는 callback
@callback(
    Output('trash', 'children'),
    Input('compare_btn', 'n_clicks'),
    State('store_exp_names', 'data')
)
def get_stored_selected_models(n, exp_names:list[str]) -> pd.DataFrame:
    global total_metrics
    global total_metrics_users
    params = {'ID':'mkdir', 'dataset_name':'ml-1m', 'exp_names': exp_names}
    response = requests.get(API_url + '/frontend/selected_metrics', params = params)  # TODO: rerank backend로 요청
    a = response.json()
    total_metrics = pd.DataFrame().from_dict(a['model_metrics'], orient='tight')
    total_metrics_users = pd.DataFrame().from_dict(a['user_metrics'], orient='tight')
    return html.Div([])


### 어떤 실험이 선택 가능한지 store에서 가져옴 (실험의 이름으로)
@callback(
    Output("selected_model_by_name", "options"),
    Input("rerank_btn", "n_clicks"),
    State("store_exp_names", "data"),
)
def print_selected_exp_name(n, exp_name):
    if n != 0:
        PreventUpdate
    exp_name = copy.deepcopy(exp_name)
    exp_name = list(set(exp_name))
    return exp_name

@callback(
    Output("reranked_graph", "children"),
    Input("rerank_btn", "n_clicks"),
    State("selected_model_by_name", "value"),
    State("alpha", "value"),
    State("obj_funcs", "value"),
    prevent_initial_update=False,
)
def plot_graph(n, model_name, alpha, obj_funcs):
    if n == 0:
        PreventUpdate
    return html.H3(str(f"{model_name}\n{alpha}\n{obj_funcs}"))
    pass