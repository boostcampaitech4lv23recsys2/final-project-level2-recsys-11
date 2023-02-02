import dash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from . import global_component as gct

dash.register_page(__name__, path='/')

img_url = "https://user-images.githubusercontent.com/76675506/216320888-7b790e97-61af-442c-93b3-c574ed0c119e.png"

feature_cn = "feature-image rounded-2 border border-primary border-4"
IMAGE_COL_WIDTH = 7
TEXT_COL_WIDTH = 5

feature1 = dbc.Row([
        dbc.Col(
            html.Img(src=img_url, className=feature_cn, ),
        width=IMAGE_COL_WIDTH),
        dbc.Col([
            html.H1("😊", className="text-end"),
            html.H3("모델의 임베딩을 다양한 관점에서 살펴볼 수 있습니다."),
            html.H5("사용자가 선택한 옵션에 따라, 임베딩 그래프를 인터랙티브하게 변화시킬 수 있습니다."),
            ], width=TEXT_COL_WIDTH)
    ], className="pt-4")

feature2 = dbc.Row([
        dbc.Col([
            html.H3("기능 소제목"),
            html.H5("사용자가 선택한 옵션에 따라, 임베딩 그래프를 인터랙티브하게 변화시킬 수 있습니다.")
        ], width=TEXT_COL_WIDTH),
        dbc.Col([
            html.Img(src=img_url, className=feature_cn),
        ], width=IMAGE_COL_WIDTH)
    ], className="feature")
 
layout = html.Div([
    gct.get_navbar(has_sidebar=False),
    html.Div([
        html.Div([
            html.H1('추천을 평가할 땐, 𝙒𝙚𝙗𝟰𝙍𝙚𝙘', className="pt-4 pb-4 text-center fs-1"),
            feature1,
            feature2,

            ], className="container"),
    ]),
])


