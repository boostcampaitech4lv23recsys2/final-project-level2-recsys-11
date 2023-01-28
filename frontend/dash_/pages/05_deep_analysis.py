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
                dcc.Dropdown(
                    options=[*uniq_genre],
                    value=['Drama'],
                    multi=True,
                    id='selected_genre'
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
                    allowCross=False,
                    id='selected_year'
                ),
                html.Br(),
            ],
            className='form-style'),
        html.Div(
            children=[
                html.P('아니면 아이템 아이디(or 제목)를 입력해보라'),
                html.P('특정 유저를 선택해서 그에 대한 추천리스트 자체를 고를 수도 있게 하자'),
                html.Hr(),
                html.Br(),
            ],
            className='form-style'),
        dcc.Store(id='item_selected', storage_type='session') #데이터를 저장하는 부분
    ],
    # className='sidebar'
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
        html.Div(
            id='side_graph'
        )
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

@callback(
    Output('item_selected', 'data'),
    Input('selected_genre', 'value'), Input('selected_year', 'value'),
)
def store_selected_item(genre, year):
    if genre is None:
        raise PreventUpdate
    tmp = item.copy()
    tmp = tmp[tmp['genre'].str.contains(
        ''.join([*map(lambda x: f'(?=.*{x})', genre)]) + '.*', regex=True)]
    tmp = tmp[(tmp['release_year'] >= year[0]) & (tmp['release_year'] <= year[1])]
    return tmp.index.to_list()

@callback(
    Output('side_graph', 'children'),
    Input('item_selected', 'data')
)
def change_side_graph(item_id):
    tmp = item.loc[item_id]
    year = px.histogram(tmp, x='release_year')
    genre = px.histogram(tmp, x='genre')
    return dcc.Graph(figure=year), dcc.Graph(figure=genre)
