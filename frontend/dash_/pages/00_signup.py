from dash import html, dcc, callback, Input, Output, State
import dash
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
from dash.exceptions import PreventUpdate

API_url = 'http://127.0.0.1:8000'

dash.register_page(__name__, path='/signup')

username = requests.get(f'{API_url}/username').json()

model_hype_type = requests.get(url=f'{API_url}/model_hype_type').json()


layout = html.Div([
    html.H1('Create User Account')
        , dcc.Location(id='create_user', refresh=True)
        , dbc.Input(id="username"
            , type="text"
            , placeholder="user name"
            , maxLength =15,
            style={'width':'20%'}),
            html.Br()
        , dbc.Input(id="password"
            , type="password"
            , placeholder="password",
            style={'width':'20%'}),
            html.Br()
        , dbc.Input(id="email"
            , type="email"
            , placeholder="email"
            , maxLength = 50,
            style={'width':'20%'}),
            html.Br()
        ,
            dbc.Button('Create User', id='submit-val', n_clicks=0, ),
            dcc.Link(
                dbc.Button('Back', n_clicks=0, style={'margin': 30}),
                href='/')
        , html.Div(id='container-button-basic'),
        
    ])#end div

@callback(
    Output(component_id='container-button-basic', component_property='children'),
    Input(component_id='submit-val', component_property='n_clicks'),
    State(component_id='username', component_property='value'),
    State(component_id='password', component_property='value'),
    State(component_id='email', component_property='value'),
)
def create_user(n_click, username, password, email):
    if n_click == 0:
        raise PreventUpdate
    # 빈칸이 있을 경우
    if (username or password or email) in ['', None]:
        return dbc.Modal([
            dbc.ModalBody("Please enter everything"),
            dbc.ModalFooter(
                dbc.Button("Close")
                )
        ], is_open=True)
    
    # insert user info into db
    pramas = {'id': username, 'password': password, 'email': email}
    response = requests.get(url=f'{API_url}/create_user', params=pramas).json()
    if response:
        return dcc.Location(id='output-location', pathname='/')
    else:
        
        return dbc.Modal([
            dbc.ModalBody("This ID already exists."),
            dbc.ModalFooter(
                dbc.Button("Close")
                )
        ], is_open=True)
    # TODO: DB에 이미 유저가 있는 경우 로그인 페이지가 아니라 경고 띄우고 그대로 있어야 함
    return f'{username} {password} {email}'