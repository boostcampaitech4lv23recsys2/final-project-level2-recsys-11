import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
import feffery_antd_components as fac



navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Compare Table", href="/compare-table")),
        dbc.NavItem(dbc.NavLink('Model vs Model', href="#")),
        dbc.NavItem(dbc.NavLink('Reranking', href="#")),
        dbc.NavItem(dbc.NavLink('Deep Anal', href="/deep_analysis"))
    ],
    brand="𝙒𝙚𝙗𝟰𝙍𝙚𝙘",
    brand_href="#",
    color="primary",
    dark=True,
    className="navbar"
)
