import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
import feffery_antd_components as fac
from . import global_component as gct

API_url = 'http://127.0.0.1:8000'

dash.register_page(__name__, path='/model-vs-model')
load_figure_template("darkly") # figure 스타일 변경

model_hype_type = requests.get(url=f'{API_url}/model_hype_type').json()

exp_df = pd.DataFrame(columns = ['model','recall','ndcg','map','popularity','colors'])

exp_df.loc[1,:] = ['M1',0.1084,0.0847,0.1011,0.0527,'red']
exp_df.loc[2,:] = ['M2',0.1124,0.0777,0.1217,0.0781,'green']
exp_df.loc[3,:] = ['M3',0.1515,0.1022,0.1195,0.0999,'blue']
exp_df.loc[4,:] = ['M4',0.0917,0.0698,0.0987,0.0315,'goldenrod']


model_form = html.Div([html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Button('➖', className='delete-btn', id='delete_button'),
            dbc.Popover("Delete this experiment", trigger='hover', target='delete_button', body=True)
]
            ),
]),
    dcc.Dropdown(list(model_hype_type.keys())),
    html.Hr(),
    html.P(
        f'''
        neg_sample: 123
        '''
    ),
], className='form-style'),
                       html.Br()])

fig_total = px.bar(
            exp_df,
            x = 'model',
            y ='recall',
            color = 'model',
            color_discrete_sequence=exp_df['colors'].values
            )
fig_ndcg = px.bar(
            exp_df,
            x = 'model',
            y ='ndcg',
            color = 'model',
            color_discrete_sequence=exp_df['colors'].values
            )
fig_map = px.bar(
            exp_df,
            x = 'model',
            y ='map',
            color = 'model',
            color_discrete_sequence=exp_df['colors'].values
            )
fig_popularity = px.bar(
            exp_df,
            x = 'model',
            y ='popularity',
            color = 'model',
            color_discrete_sequence=exp_df['colors'].values
            )

sidebar = html.Div(
    [
        html.H3("Select Expriements",),
        html.Hr(),
        html.Div(id='model_form'),
        
        dbc.Button('➕', id='add_button', n_clicks=0, style={'position':'absolute', 'right':0, 'margin-right':'2rem'}),
        dbc.Popover("Add a new expriement", trigger='hover', target='add_button', body=True),
        dbc.Button('Compare!', )
    ],
    className='sidebar'
)

button_group = html.Div(
    [
        dbc.RadioItems(
            id="radios",
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
        html.Div(id="output"),
    ],
    className="radio-group",
)

total_graph = html.Div([
    html.H1(children='Model vs Model', style={'text-align': 'center','font-weight': 'bold'}),
    html.Hr(),
    
    html.Div(id='select_model'),
    html.Div(id='select_model2'),
    

    html.H3('Total Metric'),
    dbc.Row([
      dbc.Col([
          dcc.Graph(figure=fig_total)
            ]),
            ])
                    ])

layout = html.Div(children=[
    gct.navbar,
    sidebar,
    total_graph,
    
    
],className='content')


@callback(
        Output('map', 'children'),
        Input('compare_btn', 'n_clicks'),
        prevent_initial_call=True
)
def get_quantative_metrics(form):
    params={'model_name': form['model'], 'str_key': form['values']}
    return requests.get(url=f'{API_url}/metric/quantitative/', params=params).json()[0]

@callback(
    Output('select_model2', 'children'),
    Input('uname-box', 'value'),
    prevent_initial_call=True
)
def sider_custom_trigger_demo(v):

    return v

@callback(
    Output('model_form', 'children'),
    Input('add_button', 'n_clicks')
)
def add_model_form(n):
    if n == 0:
        raise PreventUpdate
    model_form = html.Div([html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Button('➖', className='delete-btn', id=f'{n}_delete_button'),
            dbc.Popover("Delete this experiment", trigger='hover', target='delete_button', body=True)
            
]
            ),
]),
    dcc.Dropdown(list(model_hype_type.keys()), value=list(model_hype_type.keys())[0]),
    html.Hr(),
    html.P(
        f'''
        neg_sample: 123
        '''
    ),
], className='form-style'),
                       html.Br()])
    
    form_list = []
    for _ in range(n):
        form_list.append(model_form)
    return form_list

@callback(
    Output('add_button', 'n_clicks'),
    [Input('0_delete_button', 'n_clicks'), State('add_button', 'n_clicks')]
)
def delete_model_form(dn, an):
    if dn == 0:
        raise PreventUpdate
    an -=1
    return an 