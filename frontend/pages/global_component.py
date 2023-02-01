from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from pydantic import BaseSettings


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
        brand="𝙒𝙚𝙗𝟰𝙍𝙚𝙘",
        brand_href="/compare-table",
        color="primary",
        dark=True,
        className=classname
    )
    return navbar
