from dash import html, dcc, callback, Input, Output, State
import dash
import dash_bootstrap_components as dbc # pip install dash-bootstrap-components
import requests
import pandas as pd
import plotly.express as px
from dash.exceptions import PreventUpdate
import json
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
salt_value = 'zFICaaUesOyNBJW4MHuUpV'
API_url = 'http://127.0.0.1:8000'

dash.register_page(__name__, path='/signup')

username = requests.get(f'{API_url}/username').json()

model_hype_type = requests.get(url=f'{API_url}/model_hype_type').json()

error_modal = html.Div(
    dbc.Modal([
            dbc.ModalBody(id='error-modal-body'),
            dbc.ModalFooter(
                dbc.Button("Close", id='close-modal'),
                )
        ], is_open=False, id='error-modal')
)
            

layout = html.Div([
    html.H1('Create User Account')
        , dcc.Location(id='create_user', refresh=True)
        , dbc.Input(id="username"
            , type="text"
            , placeholder="Enter user name"
            , maxLength =15,
            ),
            html.Br()
        , dbc.Input(id="password1"
            , type="password"
            , placeholder="Enter password",
            ),
            html.Br()
        , dbc.Input(id="password2"
            , type="password"
            , placeholder="Confirm password",
            ),
            html.Br(),
          
            html.Div(
            id='container-button-basic'
            ),
            html.Br(),
            dbc.Button('Create User', id='submit-val', n_clicks=0, ),
            dcc.Link(
                dbc.Button('Back', n_clicks=0, style={'margin': 20}),
                href='/'),
                  ], className='login-form')

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
    
    data={'username':username, 'password1':password1, 'password2': password2,}
    response = requests.post(f'{API_url}/user/create', json=data)
    
    if response.status_code == 422:
        return dbc.Alert(response.json()['detail'][0]['msg'], color="primary"),

    elif response.status_code == 200:
        return dcc.Location(pathname='/', id='mvsm')
    
    elif response.status_code == 409:
        return dbc.Alert(response.json()['detail'], color="primary"),
    

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