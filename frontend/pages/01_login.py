import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import requests
from dash.exceptions import PreventUpdate
from passlib.context import CryptContext
from . import global_component as gct

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
salt_value = 'zFICaaUesOyNBJW4MHuUpV'

dash.register_page(__name__, path='/')


layout =  html.Div([
        html.H1('Web4Rec', style={
                                # 'padding': 10, 
                                'text-align': 'center'}),
        html.Br(),
        dcc.Location(id='url_login', refresh=True)
            , html.H5('''Please sign-in to continue:''', id='h1')
            , dbc.Input(placeholder='Enter your username',
                    type='text',
                    id='uname-box',
                    className='login-form'),
            html.Br()
            , dbc.Input(placeholder='Enter your password',
                    type='password',
                    id='pwd-box',
                    className='login-form'),
            html.Br(),
            dbc.Row([
                    dbc.Col([
                dbc.Button(children='Sign-in',
                    n_clicks=0,
                    type='submit',
                    id='login-button',
                    style={'margin':10})
            , html.Div(children='', id='output-state')]),
            dbc.Col(
            dcc.Link(
                children=dbc.Button(children='Sign-up',
                                #      style={'margin':10}
                ),
                    href='/signup'
            )),
            ]),
            html.Div(id='login-value')
            
        ], 
        style={'width':"40%"}
        ) #end div

@callback(
        Output(component_id='login-value', component_property='children'),
        Output(component_id='user_state', component_property='data'),
        
        Input('login-button', 'n_clicks'),
        State('uname-box', 'value'),
        State('pwd-box', 'value'),
        prevent_initial_call=True
)
def login(n_click, uname, pwd):
        pwd = pwd_context.hash(pwd, salt=salt_value)
        header = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {'username': uname, 'password': pwd}
        response = requests.post(f'{gct.API_URL}/user/login', data, header)
        if response.status_code == 200:
                print(response.json())
                return dcc.Location(pathname='compare-table', id='mvsm'), response.json()
        elif response.status_code == 401:
                return dbc.Alert("Invalid ID or password.", color="primary"), None
        else:
                return dbc.Alert(f"{response.status_code} Error.", color="primary"), None