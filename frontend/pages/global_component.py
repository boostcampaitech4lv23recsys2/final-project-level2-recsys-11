import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
import feffery_antd_components as fac

API_URL = 'http://127.0.0.1:30004'

def get_navbar(has_sidebar=True):
    if has_sidebar:
        classname = "navbar"
    else:
        classname = "nosidebar-navbar"
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Compare Table", href="/compare-table")),
            dbc.NavItem(dbc.NavLink('Model vs Model', href="model-vs-model")),
            dbc.NavItem(dbc.NavLink('Reranking', href="#")),
            dbc.NavItem(dbc.NavLink('Deep Anal', href="/deep_analysis_item"))
        ],
        brand="𝙒𝙚𝙗𝟰𝙍𝙚𝙘",
        brand_href="#",
        color="primary",
        dark=True,
        className=classname
    )
    return navbar
