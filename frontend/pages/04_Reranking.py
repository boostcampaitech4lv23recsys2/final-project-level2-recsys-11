import dash
from dash import html, dcc, callback, Input, Output, State,  MATCH, ALL
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
from dash.exceptions import PreventUpdate
# import feffery_antd_components as fac
from . import global_component as gct
import json
import copy

dash.register_page(__name__, path='/reranking')

exp_df = pd.DataFrame(columns = ['model','recall','ndcg','map','popularity','colors'])

exp_df.loc[1,:] = ['M1',0.1084,0.0847,0.1011,0.0527,'red']
exp_df.loc[2,:] = ['M2',0.1124,0.0777,0.1217,0.0781,'green']
exp_df.loc[3,:] = ['M3',0.1515,0.1022,0.1195,0.0999,'blue']
exp_df.loc[4,:] = ['M4',0.0917,0.0698,0.0987,0.0315,'goldenrod']

metric_list = [
    'Diversity(jaccard)',
    'Diversity(cosine)',
    'Serendipity(jaccard)',
    'Serendipity(PMI)',
    'Novelty',
]

fig_total = px.bar(
            exp_df,
            x = 'model',
            y ='recall',
            color = 'model',
            color_discrete_sequence=exp_df['colors'].values
            )
alpha_radio = html.Div([
    dbc.RadioItems(
            id="alpha",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "0.5", "value": 0.5},
                {"label": "1", "value": 1},
            ],
            value=1,
                        ),
], className="radio-group ",)
model_form = html.Div([
    html.H6("Select experiment"),
    dcc.Dropdown(id="selected_model_by_name"),
            html.Hr(),
            html.H6("Alpha: "),
            # html.P('Alpha:', className="p-0 m-0"),
            alpha_radio,
               html.H6("Select objective function with distance function"),
    dcc.Checklist(
    metric_list,
    metric_list,
    id="obj_funcs",)
], className='form-style')

sidebar = html.Div(
    [
        html.H3("Select options",),
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
          dcc.Graph(figure=fig_total, id='reranked_total_graph')
            ]),
            ])
                    ])

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
                {"label": "Qualitive", "value": 'Qual'},
                {"label": "Quantitive", "value": 'Quant'},
            ],
            value='Qual',
                        ),
            html.Br(),
            dcc.Dropdown(options=['123', '12342'], id='metric_list')
            ], width=2),
        dbc.Col([
            dcc.Graph(figure=fig_total),
            dcc.Graph(figure=fig_total),
        ], width=8)
    ]),
    ],
    className="radio-group", 
)


layout = html.Div(children=[
    gct.get_navbar(has_sidebar=False),
    html.Div([
    sidebar,
    total_graph,
    specific_metric,
    dcc.Store(id='store_exp_names', storage_type="session"),
    ])
], className="content")


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