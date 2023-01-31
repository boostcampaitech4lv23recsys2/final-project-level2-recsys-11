from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

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
            dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("User", href="/deep_analysis_user"),
                dbc.DropdownMenuItem("Item", href="/deep_analysis_item"),
            ],
            nav=True,
            in_navbar=True,
            label="Deep Analysis",
            ),
            dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Get API Key", href="#"),
                dbc.DropdownMenuItem("Logout", href="#"),
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
