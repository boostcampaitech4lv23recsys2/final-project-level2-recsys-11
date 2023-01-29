import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import requests
from dash.exceptions import PreventUpdate
import hashlib

API_url = 'http://127.0.0.1:8000'

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
                                     style={'margin':10}
                ),
                    href='/signup'
            )),
            ]),
            html.H6(id='login-value')
            
        ], style={'padding-left':'45%'}) #end div

@callback(
        Output(component_id='login-value', component_property='children'),
        Input('login-button', 'n_clicks'),
        State('uname-box', 'value'),
        State('pwd-box', 'value'),
)
def login(n_click, uname, pwd):
        if n_click == 0:
                raise PreventUpdate     
        params = {'id': uname, 'password': pwd}
        resospnse = requests.get(f'{API_url}/login_user', params=params)
        if resospnse:
                return dcc.Location(pathname='compare-table', id='mvsm')
        else:
                return dbc.Modal([
            dbc.ModalBody("Invalid ID or password."),
            dbc.ModalFooter(
                dbc.Button("Close")
                )
        ], is_open=True)
