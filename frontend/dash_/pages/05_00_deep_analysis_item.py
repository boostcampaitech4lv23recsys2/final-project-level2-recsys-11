import dash
from dash import html, dcc, callback, Input, Output, State, ctx
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
item['selected'] = 0

uniq_genre = set()
for i in item['genre']:
    uniq_genre |= set(i.split(' '))

fig = px.histogram(item, x='release_year')

fig.update_layout(clickmode='event+select')

fig.update_traces()


selection = html.Div(
    children=[
        dbc.Row(
            [dbc.Col(
                html.Div(
                    children=[
                        # 년도, 장르, 제목?, 인기도
                        html.P('장르'),
                        dcc.Dropdown(
                            options=[*uniq_genre],
                            value=[],
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
                        dcc.Store(id='items_selected_by_option', storage_type='session'), #데이터를 저장하는 부분
                        dcc.Store(id='items_selected_by_embed', storage_type='session') #데이터를 저장하는 부분
                    ],
                    className='form-style'),
            ),
            dbc.Col(
                html.Div(
                    children=[
                        html.H3('아이템 2차원 임베딩'),
                        html.P('참고로 리랭킹 관련한 지원은 유저 페이지에서만 됩니다.'),
                        html.Br(),
                        dcc.Graph(
                            id='emb_graph',
                            style={'config.responsive': True}
                        )
                    ]
                )
            ),
            dbc.Col(
                html.Div(
                    children=[
                        html.H3('사이드인포'),
                        html.Br(),
                        html.Div(id='side_graph')
                    ]
                )
            ),]
        )
    ]
)

related_users = html.Div(
    children=[
        html.H3('아이템 프로필, 아이템을 추천받은 유저'),
        html.Br(),
    ]
)


layout = html.Div(
    children=[
        gct.navbar,
        selection,
        related_users,
    ],
    # className='content'
)


# 옵션으로 선택한 아이템을 store1에 저장
@callback(
    Output('items_selected_by_option', 'data'),
    Input('selected_genre', 'value'), Input('selected_year', 'value'),
)
def save_items_selected_by_option(genre, year):
    print('storing by options')
    item_lst = item.copy()
    item_lst = item_lst[item_lst['genre'].str.contains(
        ''.join([*map(lambda x: f'(?=.*{x})', genre)]) + '.*', regex=True)]
    item_lst = item_lst[(item_lst['release_year'] >= year[0]) & (item_lst['release_year'] <= year[1])]
    return item_lst.index.to_list()

# embed graph에서 선택한 아이템을 store2에 저장
@callback(
    Output('items_selected_by_embed', 'data'),
    Input('emb_graph', 'selectedData')
)
def save_items_selected_by_embed(emb):
    print('storing by embed')
    if emb is None:
        raise PreventUpdate
    item_idx = [i['pointNumber'] for i in emb['points']]
    item_lst = item.iloc[item_idx]
    return item_lst.index.to_list()

#어느 store든 최근에 저장된 store 기준으로 임베딩과 사이드 그래프를 그림
@callback(
    Output('side_graph', 'children'),
    Output('emb_graph', 'figure'),
    Input('items_selected_by_option', 'data'),
    Input('items_selected_by_embed', 'data'),
)
def update_graph(store1, store2):
    print(ctx.triggered_id)
    if ctx.triggered_id == 'items_selected_by_option':

        tmp = item.loc[store1]
        year = px.histogram(tmp, x='release_year')
        genre = px.histogram(tmp, x='genre')

        item['selected'] = 'Not Selected'
        item.loc[store1, 'selected'] = 'Selected'
        emb = px.scatter(
            item, x = 'xs', y = 'ys', color='selected', # 갯수에 따라 색깔이 유동적인 것 같다..
            opacity=0.9,
            marginal_x="histogram",
            marginal_y="histogram",
            )
        emb.update_layout(
            uirevision="emb_graph",
            clickmode='event+select'
            )
        emb.update_traces()

        return (dcc.Graph(figure=year), dcc.Graph(figure=genre)), emb
    else:
        item['selected'] = 'Not Selected'
        if not store2:
            raise PreventUpdate
        emb = px.scatter(
            item, x = 'xs', y = 'ys', color='selected', # 갯수에 따라 색깔이 유동적인 것 같다..
            opacity=0.9,
            marginal_x="histogram",
            marginal_y="histogram",
            )
        emb.update_layout(
            datarevision="emb_graph",
            uirevision="emb_graph",
            editrevision='emb_graph',
            selectionrevision='emb_graph',
            clickmode='event+select'
            )
        # emb.update_traces()
        print(store2)

        tmp = item.loc[store2]
        year = px.histogram(tmp, x='release_year')
        genre = px.histogram(tmp, x='genre')
        return (dcc.Graph(figure=year), dcc.Graph(figure=genre)), emb


# 임베딩에서 아이템 선택 시 옵션은 비활성화
@callback(
    Output('selected_genre', 'disabled'),
    Output('selected_year', 'disabled'),
    Input('items_selected_by_embed', 'data'),
)
def disable_options(value):
    return False, False
    return True, True
