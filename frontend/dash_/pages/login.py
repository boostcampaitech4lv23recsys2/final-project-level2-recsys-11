import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import requests
API_url = 'http://127.0.0.1:8000'

dash.register_page(__name__, path='/')


layout =  html.Div([dcc.Location(id='url_login', refresh=True)
            , html.H5('''Please log in to continue:''', id='h1')
            , dcc.Input(placeholder='Enter your username',
                    type='text',
                    id='uname-box'),
            html.Br()
            , dcc.Input(placeholder='Enter your password',
                    type='password',
                    id='pwd-box'),
            html.Br()
            , html.Button(children='Sign-in',
                    n_clicks=0,
                    type='submit',
                    id='login-button',
                    style={'margin':10})
            , html.Div(children='', id='output-state'),
            html.Div(),
            dcc.Link(
                children=html.Button(children='Sign-up',
                                     style={'margin':10}
                ),
                    href='/signup'
            ),
            html.H6(id='login-value')
            
        ]) #end div

@callback(
        Output(component_id='login-value', component_property='children'),
        Input('login-button', 'n_clicks'),
        State('uname-box', 'value'),
        State('pwd-box', 'value'),
)
def login(n_click, uname, pwd):
        params = {'id': uname, 'password': pwd}
        resospnse = requests.get(f'{API_url}/login_user', params=params)
        if resospnse:
                dcc.Location('')
        return f'{uname, n_click, pwd}'