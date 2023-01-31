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

dash.register_page(__name__, path='/deep_analysis_item')
# load_figure_template("darkly") # figure 스타일 변경

user = pd.read_csv('/opt/ml/user.csv', index_col='user_id')
item = pd.read_csv('/opt/ml/item.csv', index_col='item_id')
item.fillna(value='[]', inplace=True)
item['item_profile_user'] = item['item_profile_user'].apply(eval)
item['recommended_users'] = item['recommended_users'].apply(eval)
#지금은 리스트가 담긴 행이 문자열로 저장되기에 수정하는 작업이 필요하다.
item['selected'] = 0

uniq_genre = set()
for i in item['genre']:
    uniq_genre |= set(i.split(' '))

fig = px.histogram(item, x='release_year')

fig.update_layout(clickmode='event+select')

fig.update_traces()


selection = html.Div(
    children=[
        dbc.Row([
            html.Div('유저가 장바구니에 넣은 실험들'),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem(id = '1', children="실험1",),
                    dbc.DropdownMenuItem(id = '2', children="실험2"),
                    dbc.DropdownMenuItem(id = '3', children="실험3"),],
                ),
            ]),
        dbc.Row([
            html.Div('해당 실험의 아이템, 유저 페이지'),
            dbc.ButtonGroup([
                    dbc.Button("유저", outline=True, color="primary"),
                    dbc.Button("아이템", outline=True, color="primary"),])
            ]),
        dbc.Row(
            [dbc.Col(
                html.Div(
                    children=[
                        # 년도, 장르, 제목?, 인기도
                        html.H3('옵션을 통한 선택'),
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
                        html.P('인기도'),
                        html.Br(),
                        dbc.Button(id='reset_selection', children="초기화", color="primary"),
                        dcc.Store(id='items_selected_by_option', storage_type='session'), #데이터를 저장하는 부분
                        dcc.Store(id='items_selected_by_embed', storage_type='session'), #데이터를 저장하는 부분
                        dcc.Store(id='items_for_analysis', storage_type='session'), #데이터를 저장하는 부분
                        html.P(id='n_items'),
                        dbc.Button(id='item_run',children='RUN')
                    ],
                    # className='form-style'
                ),
                width=3,

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
                    ],
                ),
                width=6,
            ),
            dbc.Col(
                html.Div(
                    children=[
                        html.H3('사이드인포'),
                        html.Br(),
                        html.Div(id='side_graph')
                    ],
                ),
                width=3,
            ),]
        )
    ]
)

#그림 카드 만드는 함수
def make_card(num):
    card = dbc.Col(
        id=f'item_{num}',

    )
    return card

top = html.Div(
    children=[
        html.H3('top pop 10'),
        dbc.Row(id='top_pop_10',),
        html.H3('top rec 10'),
        dbc.Row(id='top_rec_10',),
        html.Br(),
        html.P(id='test')
    ]
)

related_users = html.Div(
    children=[
        html.H3('유저 프로필, 유저 추천 리스트'),
        html.Br(),
    ]
)


layout = html.Div(
    children=[
        gct.get_navbar(has_sidebar=False),
        selection,
        top,
        related_users
    ],
    # className='content'
)


# 옵션으로 선택한 아이템을 store1에 저장
@callback(
    Output('items_selected_by_option', 'data'),
    Input('selected_genre', 'value'), Input('selected_year', 'value'),
)
def save_items_selected_by_option(genre, year):
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
    if emb is None:
        raise PreventUpdate
    item_idx = [i['pointNumber'] for i in emb['points']]
    item_lst = item.iloc[item_idx]
    return item_lst.index.to_list()

# 최근에 선택한 아이템을 최종 store에 저장
@callback(
    Output('items_for_analysis', 'data'),
    Output('n_items', 'children'),
    Input('items_selected_by_option', 'data'),
    Input('items_selected_by_embed', 'data'),
)
def prepare_analysis(val1, val2):
    if ctx.triggered_id == 'items_selected_by_option':
        return val1, f'selected items: {len(val1)}'
    else:
        return val2, f'selected items: {len(val2)}'


#최근에 저장된 store 기준으로 임베딩 그래프를 그림
@callback(
    Output('emb_graph', 'figure'),
    Input('items_selected_by_option', 'data'),
)
def update_graph(store1):
    item['selected'] = 'Not Selected'
    item.loc[store1, 'selected'] = 'Selected'
    emb = px.scatter(
        item, x = 'xs', y = 'ys', color='selected', # 갯수에 따라 색깔이 유동적인 것 같다..
        opacity=0.9,
        marginal_x="histogram",
        marginal_y="histogram",
    )
    emb.update_layout(clickmode='event+select')
    return emb

#최근에 저장된 store 기준으로 사이드 그래프를 그림
@callback(
    Output('side_graph', 'children'),
    Input('items_selected_by_option', 'data'),
    Input('items_selected_by_embed', 'data'),
)
def update_graph(store1, store2):
    if ctx.triggered_id == 'items_selected_by_option':
        tmp = item.loc[store1]
        year = px.histogram(tmp, x='release_year')
        genre = px.histogram(tmp, x='genre')
        return (dcc.Graph(figure=year), dcc.Graph(figure=genre))
    else:
        if not store2:
            raise PreventUpdate
        tmp = item.loc[store2]
        tmp = tmp[tmp['selected'] == 'Selected']
        year = px.histogram(tmp, x='release_year')
        genre = px.histogram(tmp, x='genre')
        return (dcc.Graph(figure=year), dcc.Graph(figure=genre))


# 초기화 버튼 누를 때 선택 초기화
@callback(
    Output('selected_genre', 'value'),
    Output('selected_year', 'value'),
    Output('item_run', 'n_clicks'),
    Input('reset_selection', 'n_clicks'),
)
def reset_selection(value):
    return [], [item['release_year'].min(), item['release_year'].max()], 0

#### run 실행 시 실행될 함수들 #####

# 테스팅
@callback(
    Output('test', 'children'),
    Input('item_run', 'n_clicks'),
    State('items_for_analysis', 'data'),
    prevent_initial_call=True
)
def prepare_analysis(value, data):
    if value != 1:
        raise PreventUpdate
    else:
        return data

# top pop 10
@callback(
    Output('top_pop_10', 'children'),
    Input('item_run', 'n_clicks'),
    State('items_for_analysis', 'data'),
    prevent_initial_call=True
)
def draw_toppop_card(value, data):
    if value != 1:
        raise PreventUpdate
    else:
        def make_card(element):
            tmp = item.loc[element]
            card = dbc.Col(
                children=dbc.Card([
                    dbc.CardImg(top=True),
                    dbc.CardBody([
                        html.H6(tmp['movie_title']),
                        html.P(tmp['genre']),
                        html.P(tmp['release_year']),
                        html.P(tmp['item_pop']),
                    ],),
                ],),
            )
            return card
        pop = item.loc[data].sort_values(by=['item_pop'], ascending=False).head(10).index
        lst = [make_card(item) for item in pop] # 보여줄 카드 갯수 지정 가능
        return lst
    
# top rec 10
@callback(
    Output('top_rec_10', 'children'),
    Input('item_run', 'n_clicks'),
    State('items_for_analysis', 'data'),
    prevent_initial_call=True
)
def draw_toprec_card(value, data):
    if value != 1:
        raise PreventUpdate
    else:
        def make_card(element):
            tmp = item.loc[element]
            card = dbc.Col(
                children=dbc.Card([
                    dbc.CardImg(top=True),
                    dbc.CardBody([
                        html.H6(tmp['movie_title']),
                        html.P(tmp['genre']),
                        html.P(tmp['release_year']),
                        html.P(tmp['item_pop']),
                    ],),
                ],),
            )
            return card
        
        item['len']  = item['recommended_users'].apply(len)
        
        rec = item.loc[data].sort_values(by=['len'], ascending=False).head(10).index
        lst = [make_card(item) for item in rec] # 보여줄 카드 갯수 지정 가능
        
        item.drop('len', axis=1, inplace=True)
        return lst

#pd.DataFrame.from_dict(data=data, orient='tight')데이터프래임
# request.get().json
# 저장
