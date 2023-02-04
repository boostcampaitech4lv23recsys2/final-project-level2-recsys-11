from dash import html, dcc, callback, Input, Output, State
import dash
import dash_bootstrap_components as dbc # pip install dash-bootstrap-components
import requests
import pandas as pd
import plotly.express as px
from dash.exceptions import PreventUpdate
import json
from passlib.context import CryptContext
from . import global_component as gct

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
salt_value = gct.get_login_setting()['SALT']
API_url = 'http://127.0.0.1:30004'

dash.register_page(__name__, path='/signup')


layout = html.Div([
    html.H1('회원가입')
        , dcc.Location(id='create_user', refresh=True)
        , dbc.Input(id="username"
            , type="text"
            , placeholder="아이디"
            , maxLength =15,
            ),
            html.Br()
        , dbc.Input(id="password1"
            , type="password"
            , placeholder="비밀번호",
            ),
            html.Br()
        , dbc.Input(id="password2"
            , type="password"
            , placeholder="비밀번호 확인",
            ),
            html.Br(),
          
            html.Div(
            id='container-button-basic'
            ),
            html.Br(),
            dbc.Button('회원가입', id='submit-val', n_clicks=0, ),
            dcc.Link(
                dbc.Button('뒤로가기', n_clicks=0, style={'margin': 20}),
                href='/'),
                  ], className="mx-auto w-25")

@callback(
    Output(component_id='container-button-basic', component_property='children'),
    
    Input(component_id='submit-val', component_property='n_clicks'),
    State(component_id='username', component_property='value'),
    State(component_id='password1', component_property='value'),
    State(component_id='password2', component_property='value'),

    prevent_initial_call=True
)
def create_user(n_click, username, password1, password2):
    # hashing password
    password1=pwd_context.hash(password1, salt=salt_value)
    password2=pwd_context.hash(password2, salt=salt_value)
    
    data={'ID':username, 'password1':password1, 'password2': password2,}
    response = requests.post(f'{API_url}/user/create_user', json=data)
    
    if response.status_code == 422:
        return dbc.Alert("Password doesn't match. Please check agian.", color="primary"),

    elif response.status_code == 200:
        return dcc.Location(pathname='/', id='mvsm')
    
    elif response.status_code == 409:
        return dbc.Alert("Already exist ID.", color="primary"),
    

@callback(
    Output('password2', 'valid'),
    Output('password2', 'invalid'),
    Input('password2', 'value'),
    State('password1', 'value'),
    prevent_initial_call=True
)
def is_same_password(password2, password1):
    password1=pwd_context.hash(password1, salt=salt_value)
    password2=pwd_context.hash(password2, salt=salt_value)
    if password2 == password1:
        return True, False
    else:
        return False, True