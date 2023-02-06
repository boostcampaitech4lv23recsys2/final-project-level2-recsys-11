from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from pydantic import BaseSettings
import requests
from dash.exceptions import PreventUpdate

env_path = '/opt/ml/final-project-level2-recsys-11/backend/login.env'
class Login_Settings(BaseSettings):
        SALT: str
        ACCESS_TOKEN_EXPIRE_MINUTES: int
        SECRET_KEY: str
        ALGORITHM: str

        class config:
                env_flie = '.env'
                env_flie_encoding = 'utf-8'

def get_login_setting():
    return Login_Settings(_env_file=env_path, _env_file_encoding='utf-8').dict()

API_URL = 'http://127.0.0.1:30004'

BRAND_LOGO = "𝙒𝙚𝙗𝟰𝙍𝙚𝙘"

def get_navbar(has_sidebar=True):
    if has_sidebar:
        classname = "navbar"
    else:
        classname = "nosidebar-navbar"
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Compare Table", href="/compare-table")),
            dbc.NavItem(dbc.NavLink('Model vs Model', href="/model-vs-model")),
            dbc.NavItem(dbc.NavLink('Reranking', href="/reranking")),
            dbc.NavItem(dbc.NavLink('Deep Analysis', href="/deep_analysis")),
            dbc.NavItem(dbc.NavLink('FAQ', href="/FAQ")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Get API Key", href="#"),
                    dbc.DropdownMenuItem("Logout", href="/login"),
                    html.Hr(),
                    dbc.DropdownMenuItem("About", href="#"),
                ],
                nav=True,
                in_navbar=True,
                label="Settings",
            ),
        ],
        brand=BRAND_LOGO,
        brand_href="/",
        brand_style={'color':'#FFFAF0'},
        color="primary",
        dark=True,
        sticky="top",
        className=classname + " p-0 sticky",
    #     style={"position": "fixed",
    #            "top": 0,
    #            "width": "100%",}
    )
    return navbar

def Authenticate(input_btn_name:str, output_component_name:str):
    @callback(
        Output(f'{output_component_name}', 'options'),
        Input(f'{input_btn_name}', 'n_clicks'),
        State('store_user_state', 'data')
    )
    def get_dataset_list(n, user_state):
        if n != 0:
            PreventUpdate
        response = requests.post(f"{API_URL}/user/get_current_user", json=user_state)
        if response.status_code == 201:
            return [1,2,3,4]
        else:
            return list(str(response))