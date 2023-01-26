import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
import feffery_antd_components as fac
from . import global_component as gct

API_url = 'http://127.0.0.1:8000'

dash.register_page(__name__, path='/deep_analysis')
# load_figure_template("darkly") # figure 스타일 변경

user = pd.read_csv('/opt/ml/user.csv', index_col='user_id')
item = pd.read_csv('/opt/ml/item.csv', index_col='item_id')

uniq_genre = set()
for i in item['genre']:
    uniq_genre |= set(i.split(' '))

fig = px.histogram(item, x='release_year')

fig.update_layout(clickmode='event+select')

fig.update_traces()

sidebar = html.Div(
    children=[
        html.Div(
            children=[
                html.P('다른 페이지에서 입력한 아이템'),
                html.P('가령 유저 페이지에서 20대가 본 아이템들에 대한 분석을 하고 싶어서 넘어온 상황!'),
                html.Button('취소!'),
                html.P('그러면 아래 블록이 나옴(이 블록 사라지면서)'),
                html.Br(),
            ],
            className='form-style'),
        html.Div(
            children=[
                html.P('아니면 니가 선별해서 골라보라'),
                html.Hr(),
                # 년도, 장르, 제목?, 인기도
                html.P('장르'),
                dcc.Checklist(
                    options=[*uniq_genre],
                ),
                html.P('년도'),
                dcc.RangeSlider(
                    min=item['release_year'].min(),
                    max=item['release_year'].max(),
                    value=[item['release_year'].min(), item['release_year'].max()],
                    step=1,
                    marks={
                        item['release_year'].min():str(item['release_year'].min()),
                        item['release_year'].max():str(item['release_year'].max()),
                    },
                    tooltip={"placement": "bottom", "always_visible": True},
                    allowCross=False
                ),
                html.Br(),
            ],
            className='form-style'),
        html.Div(
            children=[
                html.P('아니면 아이템 아이디(or 제목)를 입력해보라'),
                html.Hr(),
                html.Br(),
            ],
            className='form-style'),
    ],
    className='sidebar'
)

embedding = html.Div(
    children=[
        html.H3('아이템 2차원 임베딩'),
        html.P('참고로 리랭킹 관련한 지원은 유저 페이지에서만 됩니다.'),
        html.Br()
    ]
)

side = html.Div(
    children=[
        html.H3('사이드인포'),
        html.Br(),
        dcc.Graph(
            id='basic-interactions',
            figure=fig,
        ),
    ]
)

related_users = html.Div(
    children=[
        html.H3('아이템 프로필, 아이템을 추천받은 유저'),
        html.Br(),
    ]
)

layout = html.Div(children=[
    gct.navbar,
    sidebar,
    embedding,
    side,
    related_users,
],className='content')