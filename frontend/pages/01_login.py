import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import requests
from dash.exceptions import PreventUpdate
from passlib.context import CryptContext
from . import global_component as gct
from pydantic import BaseSettings


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
salt_value = gct.get_login_setting()['SALT']

dash.register_page(__name__, path='/login')

layout =  html.Div([
        # html.Br(),
        # html.Br(),
        # html.Br(),
        # html.Br(),
        html.Br(),
   
        html.H1('Web4Rec', style={
                                # 'padding': 10,
                                'text-align': 'center'}),
        html.Hr(),
        html.Br(),
        html.H5('로그인', style={'text-align': 'center'}),
        html.Hr(style={'width': '50%', 'margin-left': '25%'}),
        dcc.Location(id='url_login', refresh=True)
            # html.H5('''가입한 아이디로 로그인 해주세요''', id='h1')
            , html.Br() 
            , dbc.Input(placeholder='아이디',
                    type='text',
                    id='uname-box',
                    className='login-form'),
            html.Br()
            , dbc.Input(placeholder='비밀번호',
                    type='password',
                    id='pwd-box',
                    className='login-form'),
            html.Br(),
            dbc.Row([
                    dbc.Col([
                        dbc.Button(children='로그인',
                                n_clicks=0,
                                type='submit',
                                id='login-button',
                                className='w-100'
                #   style={'margin':10},
                                ),
                        html.Div(children='', id='output-state')]),
                    dbc.Col(
                        dcc.Link(
                        children=dbc.Button(children='회원가입', className='w-100', color="secondary"
                                #      style={'margin':10}
                ),
                    href='/signup',
            )),
            ]),
            html.Div(id='login-value')

        ],
        # style={'width':"40%"},
        className="mx-auto w-25 mt-5"
        ) #end div

@callback(
        Output(component_id='login-value', component_property='children'),
        Output(component_id='store_user_state', component_property='data'),

        Input('login-button', 'n_clicks'),
        State('uname-box', 'value'),
        State('pwd-box', 'value'),
        prevent_initial_call=True
)
def login(n_click, uname, pwd):
        if n_click == 0:
                return None, None
        try:
                pwd = pwd_context.hash(pwd, salt=salt_value)
        except TypeError as e:
                pass
        header = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {'username': uname, 'password': pwd}
        response = requests.post(f'{gct.API_URL}/user/login', data, header)
        if response.status_code == 200:
                return dcc.Location(pathname='compare-table', id='mvsm'), response.json()
        elif response.status_code == 401:
                return dbc.Alert("Invalid ID or password.", color="primary"), None
        else:
                return dbc.Alert(f"{response.status_code} Error.", color="primary"), None