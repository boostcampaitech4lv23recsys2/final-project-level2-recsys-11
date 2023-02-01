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
from collections import Counter
from plotly.subplots import make_subplots
import plotly.graph_objects as go


dash.register_page(__name__, path='/deep_analysis')
# load_figure_template("darkly") # figure 스타일 변경

user = pd.read_csv('/opt/ml/user.csv', index_col='user_id')
item = pd.read_csv('/opt/ml/item.csv', index_col='item_id')
item.fillna(value='[]', inplace=True)
item['item_profile_user'] = item['item_profile_user'].apply(eval)
item['recommended_users'] = item['recommended_users'].apply(eval)
#리스트와 같은 객체는 json으로 넘어올 때 문자열로 들어올 가능성이 있으니 이 코드가 필요할 수도 있다. 상황에 따라 판단하기.
item['selected'] = 0
item['len']  = item['recommended_users'].apply(len)

uniq_genre = set()
for i in item['genre']:
    uniq_genre |= set(i.split(' '))

#잠시 리랭크는 내가 임의로 진행한다.
tmp_user = [1, 5, 10]
obj = 'novelty'
cand = user.loc[tmp_user]
#이걸 가지고 백엔드에 소통할 것이다.

base = html.Div(
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
            dbc.RadioItems(
                id="show_user_or_item",
                # className="btn-group",
                # inputClassName="btn-check",
                # labelClassName="btn btn-outline-primary",
                # labelCheckedClassName="active",
                options=[
                    {"label": "item", "value": 1},
                    {"label": "user", "value": 2},
                ],
                value=1,
            ),
        ]),
    ]
)


item_selection = html.Div(
    children=[
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
                        dbc.Button(id='item_reset_selection', children="초기화", color="primary"),
                        html.P(id='n_items'),
                        dbc.Button(id='item_run',children='RUN'),
                        dcc.Store(id='items_selected_by_option', storage_type='session'), #데이터를 저장하는 부분
                        dcc.Store(id='items_selected_by_embed', storage_type='session'), #데이터를 저장하는 부분
                        dcc.Store(id='items_for_analysis', storage_type='session'), #데이터를 저장하는 부분
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
                            id='item_emb_graph',
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
                        html.Div(id='item_side_graph'),
                    ],
                    style={'overflow': 'scroll', 'height':700}
                ),
                width=3,
            ),]
        )
    ]
)

item_top = html.Div(
    children=[
        html.H3('top pop 10'),
        dbc.Row(id='top_pop_10',),
        html.H3('top rec 10'),
        dbc.Row(id='top_rec_10',),
        html.Br(),
    ]
)

item_related_users = html.Div(
    children=[
        html.H3('유저 프로필, 유저 추천 리스트'),
        dbc.Row([
            dbc.Col(id='related_user_age'),
            dbc.Col(id='related_user_gender'),
            dbc.Col(id='related_user_occupation'),
        ]),
        html.Br(),
    ]
)


user_selection = html.Div(
    children=[
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        children=[
                            html.H3("옵션을 통한 선택"),
                            html.P("연령대"),
                            dcc.Checklist(
                                options=sorted(user["age"].unique()),
                                value=[],
                                id="selected_age",
                            ),
                            html.P("성별"),
                            dcc.Dropdown(
                                options=["M", "F"],
                                value=[],
                                id="selected_gender",
                            ),
                            html.P("직업"),
                            dcc.Dropdown(
                                options=sorted(user["occupation"].unique()),
                                value=[],
                                multi=True,
                                id="selected_occupation",
                            ),
                            dcc.Checklist(
                                options=['틀린 유저들만(0.5 기준으로)'],
                                value=[],
                                id="selected_wrong",
                            ),
                            html.Br(),
                            dbc.Button(
                                id="user_reset_selection", children="초기화", color="primary"
                            ),
                            html.P(id="n_users"),
                            dbc.Button(id="user_run", children="RUN"),
                            dcc.Store(id="users_selected_by_option", storage_type="session"),  # 데이터를 저장하는 부분
                            dcc.Store(id="users_selected_by_embed", storage_type="session"),  # 데이터를 저장하는 부분
                            dcc.Store(id="users_for_analysis", storage_type="session"),  # 데이터를 저장하는 부분
                        ],
                        # className='form-style'
                    ),
                    width=3,
                ),
                dbc.Col(
                    html.Div(
                        children=[
                            html.H3("유저 2차원 임베딩"),
                            dcc.Graph(id="user_emb_graph", style={"config.responsive": True}),
                        ],
                    ),
                    width=6,
                ),
                dbc.Col(
                    html.Div(
                        children=[
                            html.H3("사이드인포"),
                            html.Div(id="user_side_graph"),
                        ],
                        style={'overflow': 'scroll', 'height':700}
                    ),
                    width=3,
                ),
            ]
        ),
    ]
)

user_rerank = dbc.Row(
    id='rerank_box',
)

user_analysis = html.Div(
    id = 'user_deep_analysis'
)

layout = html.Div(
    children=[
        gct.get_navbar(has_sidebar=False),
        html.Div(
            children=[
                base,
                html.Div(id='deep_analysis_page',)
            ],
            className='container'
        )
    ],
)


test = html.Div(html.P(id='testin'))

# 유저 페이지를 띄울지, 아이템 페이지를 띄울지
@callback(
    Output('deep_analysis_page', 'children'),
    Input('show_user_or_item','value')
)
def display_overall(val):
    if val == 1:
        return [
            item_selection,
            item_top,
            item_related_users,
        ]
    else:
        return [
            user_selection,
            user_rerank,
            user_analysis,
            test,

        ]

#########################################################
######################## 아이템 ##########################
#########################################################

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
    Input('item_emb_graph', 'selectedData')
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
    Output('item_emb_graph', 'figure'),
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
    emb.update_layout(
        clickmode='event+select',
        width=700,
        height=700,
    )
    return emb

#최근에 저장된 store 기준으로 사이드 그래프를 그림
@callback(
    Output('item_side_graph', 'children'),
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
    Input('item_reset_selection', 'n_clicks'),
)
def item_reset_selection(value):
    return [], [item['release_year'].min(), item['release_year'].max()], 0

#### run 실행 시 실행될 함수들 #####

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
        rec = item.loc[data].sort_values(by=['len'], ascending=False).head(10).index
        lst = [make_card(item) for item in rec] # 보여줄 카드 갯수 지정 가능
        return lst


# 관련 유저 그래프 시각화
@callback(
    Output('related_user_age', 'children'),
    Output('related_user_gender', 'children'),
    Output('related_user_occupation', 'children'),
    Input('item_run', 'n_clicks'),
    State('items_for_analysis', 'data'),
    prevent_initial_call=True
)
def draw_user_graph(value, data):
    if value != 1:
        raise PreventUpdate
    else:

        def get_user_side_by_items(selected_item: list) -> tuple:
            '''
            선택된 item들의 idx를 넣어주면, 그 아이템들을 사용한 유저, 추천받은 유저들의 인구통계학적 정보 수집
            총 6개의 Counter가 return, 앞에서 부터 2개씩 age, gender, occupation 정보
            e.g., 앞의 age는 사용한 유저, 뒤의 age는 추천받은 유저들 ...
            '''
            # Counter 세팅
            age_Counter_profile, gender_Counter_profile, occupation_Counter_profile = Counter(), Counter(), Counter()
            age_Counter_rec, gender_Counter_rec, occupation_Counter_rec = Counter(), Counter(), Counter()

            for idx in selected_item:
                one_item = item.loc[idx]

                # profile Counter
                tmp = user.loc[one_item['item_profile_user'], ['age','gender','occupation']]
                age_Counter_profile += Counter(tmp['age'])
                gender_Counter_profile += Counter(tmp['gender'])
                occupation_Counter_profile += Counter(tmp['occupation'])

                # profile Counter
                if one_item.isnull()['recommended_users']:
                    continue
                tmp = user.loc[one_item['recommended_users'], ['age','gender','occupation']]
                age_Counter_rec += Counter(tmp['age'])
                gender_Counter_rec += Counter(tmp['gender'])
                occupation_Counter_rec += Counter(tmp['occupation'])

            age_Counter_profile = dict(sorted(age_Counter_profile.items(), key=lambda x : x[1], reverse=True))
            age_Counter_rec = dict(sorted(age_Counter_rec.items(), key=lambda x : x[1], reverse=True))

            gender_Counter_profile = dict(sorted(gender_Counter_profile.items(), key=lambda x : x[1], reverse=True))
            gender_Counter_rec = dict(sorted(gender_Counter_rec.items(), key=lambda x : x[1], reverse=True))

            occupation_Counter_profile = dict(sorted(occupation_Counter_profile.items(), key=lambda x : x[1], reverse=True))
            occupation_Counter_rec = dict(sorted(occupation_Counter_rec.items(), key=lambda x : x[1], reverse=True))

            return age_Counter_profile, age_Counter_rec, gender_Counter_profile, gender_Counter_rec, occupation_Counter_profile, occupation_Counter_rec


        def plot_age_counter(age_Counter_profile: Counter, age_Counter_rec: Counter):
            age_Counter_profile_labels = list(age_Counter_profile.keys())
            age_Counter_profile_values = list(age_Counter_profile.values())
            age_Counter_rec_labels = list(age_Counter_rec.keys())
            age_Counter_rec_values = list(age_Counter_rec.values())
            fig = make_subplots(
                rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                subplot_titles=("Age rtio(profile)", "Age ratio(rec)")
            )
            fig.add_trace(go.Pie(labels=age_Counter_profile_labels, values=age_Counter_profile_values, name="Age(profile)", pull=[0.07]+[0]*(len(age_Counter_profile_values)-1)), # textinfo='label+percent', pull=[0.2]+[0]*(len(total_item_genre_values)-1)
                    1, 1)
            fig.add_trace(go.Pie(labels=age_Counter_rec_labels, values=age_Counter_rec_values, name="Age(rec)", pull=[0.07]+[0]*(len(age_Counter_rec_values)-1)), # textinfo='label+percent', pull=[0.2]+[0]*(len(user_profile_values)-1)
                    1, 2)

            fig.update_traces(hole=0.3, hoverinfo="label+percent+name")
            fig.update_layout(
                title_text="Selected Item profile vs Selected Item rec list (Age)",
            #     width=1000,
            #     height=500
            )

            return fig

        def plot_gender_counter(gender_Counter_profile: Counter, gender_Counter_rec: Counter):
            gender_Counter_profile_labels = list(gender_Counter_profile.keys())
            gender_Counter_profile_values = list(gender_Counter_profile.values())
            gender_Counter_rec_labels = list(gender_Counter_rec.keys())
            gender_Counter_rec_values = list(gender_Counter_rec.values())

            fig = make_subplots(
                rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                subplot_titles=("Gender ratio(profile)", "Gender ratio(rec)")
            )
            fig.add_trace(go.Pie(labels=gender_Counter_profile_labels, values=gender_Counter_profile_values, name="user Rec list genre", pull=[0.07]+[0]*(len(gender_Counter_profile_values)-1)), # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rec_values)-1)
                    1, 1)
            fig.add_trace(go.Pie(labels=gender_Counter_rec_labels, values=gender_Counter_rec_values, name="user rerank", pull=[0.07]+[0]*(len(gender_Counter_rec_values)-1)), # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rerank_values)-1)
                    1, 2)
            fig.update_traces(hole=0.3, hoverinfo="label+percent+name")
            fig.update_layout(
                title_text="Selected Item profile vs Selected Item rec list (Gender)",
                # width=1000,
                # height=500
            )

            return fig

        def plot_occupation_counter(occupation_Counter_profile: Counter, occupation_Counter_rec: Counter):
            occupation_Counter_profile_labels = list(occupation_Counter_profile.keys())
            occupation_Counter_profile_values = list(occupation_Counter_profile.values())
            occupation_Counter_rec_labels = list(occupation_Counter_rec.keys())
            occupation_Counter_rec_values = list(occupation_Counter_rec.values())
            fig = make_subplots(
                rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                subplot_titles=("Occupation ratio(profile)", "Occupation ratio(rec)")
            )
            fig.add_trace(go.Pie(labels=occupation_Counter_profile_labels, values=occupation_Counter_profile_values, name="user Rec list genre", pull=[0.07]+[0]*(len(occupation_Counter_profile_values)-1)), # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rec_values)-1)
                    1, 1)
            fig.add_trace(go.Pie(labels=occupation_Counter_rec_labels, values=occupation_Counter_rec_values, name="user rerank", pull=[0.07]+[0]*(len(occupation_Counter_rec_values)-1)), # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rerank_values)-1)
                    1, 2)
            fig.update_traces(hole=0.3, hoverinfo="label+percent+name")
            fig.update_layout(
                title_text="Selected Item profile vs Selected Item rec list (Occupation)",
                # width=1000,
                # height=500
            )

            return fig

        age_Counter_profile, age_Counter_rec, gender_Counter_profile, gender_Counter_rec, occupation_Counter_profile, occupation_Counter_rec = get_user_side_by_items(data)
        age = dcc.Graph(figure=plot_age_counter(age_Counter_profile, age_Counter_rec))
        gender = dcc.Graph(figure=plot_gender_counter(gender_Counter_profile, gender_Counter_rec))
        occupation = dcc.Graph(figure=plot_occupation_counter(occupation_Counter_profile, occupation_Counter_rec))

        return age, gender, occupation



#########################################################
######################## 유저 ###########################
#########################################################

# @callback(Output('testin', 'children'), Input('users_selected_by_option', 'data'))
# def testing(val):
#     tmp = [*map(str, val)]
#     return str(tmp)

#선택한 유저들을 store1에 저장
@callback(
    Output('users_selected_by_option', 'data'),
    Input('selected_age', 'value'),
    Input('selected_gender', 'value'),
    Input('selected_occupation', 'value'),
    Input('selected_wrong', 'value'),
)
def save_users_selected_by_option(age, gender, occupation, wrong):
    user_lst = user.copy()
    if age:
        user_lst = user_lst[user_lst['age'].isin(age)]
    if gender:
        user_lst = user_lst[user_lst['gender']==gender]
    if occupation:
        user_lst = user_lst[user_lst['occupation'].isin(occupation)]
    if wrong:
        user_lst = user_lst[user_lst['div_jac'] <= 0.77] # 당장의 데이터에 recall이 없다.
    return user_lst.index.to_list()

# user embed graph에서 선택한 유저들을 store2에 저장
@callback(
    Output('users_selected_by_embed', 'data'),
    Input('user_emb_graph', 'selectedData')
)
def save_users_selected_by_embed(emb):
    if emb is None:
        raise PreventUpdate
    user_idx = [i['pointNumber'] for i in emb['points']]
    user_lst = user.iloc[user_idx]
    return user_lst.index.to_list()

# 최근에 선택한 유저를 최종 store에 저장
@callback(
    Output('users_for_analysis', 'data'),
    Output('n_users', 'children'),
    Input('users_selected_by_option', 'data'),
    Input('users_selected_by_embed', 'data'),
)
def prepare_analysis(val1, val2):
    if ctx.triggered_id == 'users_selected_by_option':
        return val1, f'selected users: {len(val1)}'
    else:
        return val2, f'selected users: {len(val2)}'

#최근에 저장된 store 기준으로 유저 임베딩 그래프를 그림
# @callback(
#     Output('user_emb_graph', 'figure'),
#     Input('users_selected_by_option', 'data'),
# )
# def update_graph(store1):
#     user['selected'] = 'Not Selected'
#     user.loc[store1, 'selected'] = 'Selected'
#     emb = px.scatter(
#         user, x = 'xs', y = 'ys', color='selected', # 갯수에 따라 색깔이 유동적인 것 같다..
#         opacity=0.9,
#         marginal_x="histogram",
#         marginal_y="histogram",
#     )
#     emb.update_layout(
#         clickmode='event+select',
#         width=700,
#         height=700,
#     )
#     return emb

# #최근에 저장된 store 기준으로 사이드 그래프를 그림
@callback(
    Output('user_side_graph', 'children'),
    Input('users_selected_by_option', 'data'),
    Input('users_selected_by_embed', 'data'),
)
def update_graph(store1, store2):
    def plot_age_counter(age_Counter_profile: Counter):
        age_Counter_profile_labels = list(age_Counter_profile.keys())
        age_Counter_profile_values = list(age_Counter_profile.values())
        fig = make_subplots(
            rows=1, cols=1, specs=[[{'type':'domain'}]],
            subplot_titles=("Age ratio(profile)")
        )
        fig.add_trace(go.Pie(labels=age_Counter_profile_labels, values=age_Counter_profile_values, name="Age(profile)", pull=[0.07]+[0]*(len(age_Counter_profile_values)-1)), # textinfo='label+percent', pull=[0.2]+[0]*(len(total_item_genre_values)-1)
                1, 1)

        fig.update_traces(hole=0.3, hoverinfo="label+percent+name")
        return fig

    def plot_gender_counter(gender_Counter_profile: Counter):
        gender_Counter_profile_labels = list(gender_Counter_profile.keys())
        gender_Counter_profile_values = list(gender_Counter_profile.values())

        fig = make_subplots(
            rows=1, cols=1, specs=[[{'type':'domain'}]],
            subplot_titles=("Gender ratio(profile)")
        )
        fig.add_trace(go.Pie(labels=gender_Counter_profile_labels, values=gender_Counter_profile_values, name="user Rec list genre", pull=[0.07]+[0]*(len(gender_Counter_profile_values)-1)), # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rec_values)-1)
                1, 1)
        fig.update_traces(hole=0.3, hoverinfo="label+percent+name")

        return fig

    def plot_occupation_counter(occupation_Counter_profile: Counter):
        occupation_Counter_profile_labels = list(occupation_Counter_profile.keys())
        occupation_Counter_profile_values = list(occupation_Counter_profile.values())
        fig = make_subplots(
            rows=1, cols=1, specs=[[{'type':'domain'}]],
            subplot_titles=("Occupation ratio(profile)")
        )
        fig.add_trace(go.Pie(labels=occupation_Counter_profile_labels, values=occupation_Counter_profile_values, name="user Rec list genre", pull=[0.07]+[0]*(len(occupation_Counter_profile_values)-1)), # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rec_values)-1)
                1, 1)
        fig.update_traces(hole=0.3, hoverinfo="label+percent+name")
        return fig

    if ctx.triggered_id == 'users_selected_by_option':
        tmp = user.loc[store1]
        age = plot_age_counter(Counter(tmp['age']))
        gender = plot_gender_counter(Counter(tmp['gender']))
        occupation = plot_occupation_counter(Counter(tmp['occupation']))
        return (dcc.Graph(figure=age), dcc.Graph(figure=gender), dcc.Graph(figure=occupation))
    else:
        if not store2:
            raise PreventUpdate
        tmp = user.loc[store2]
        tmp = tmp[tmp['selected'] == 'Selected']
        age = plot_age_counter(Counter(tmp['age']))
        gender = plot_gender_counter(Counter(tmp['gender']))
        occupation = plot_occupation_counter(Counter(tmp['occupation']))
        return (dcc.Graph(figure=age), dcc.Graph(figure=gender), dcc.Graph(figure=occupation))

# 초기화 버튼 누를 때 선택 초기화
@callback(
    Output('selected_age', 'value'),
    Output('selected_gender', 'value'),
    Output('selected_occupation', 'value'),
    Output('selected_wrong', 'value'),
    Output('user_run', 'n_clicks'),
    Input('user_reset_selection', 'n_clicks'),
)
def item_reset_selection(value):
    return [], [], [], [], 0


#### run 실행 시 실행될 함수들 #####

#최종 store가 업데이트됐을 때 리랭킹 선택지 등장
@callback(
    Output('rerank_box', 'children'),
    Input('user_run', 'n_clicks'),
    prevent_initial_call=True
)
def prepare_rerank(value):
    if value != 1:
        raise PreventUpdate
    else:
        tmp = [
            html.P("Reranking Options  a *rel(i) + (1-a) * obj(i)"),
            html.Br(),
            dbc.Col(
                dbc.RadioItems(
                    id="rerank_obj",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-primary",
                    labelCheckedClassName="active",
                    options=[
                        {"label": "Diversity(Cosine)", "value": 1},
                        {"label": "Diversity(Jaccard)", "value": 2},
                        {"label": "Serendipity(PMI)", "value": 3},
                        {"label": "Serendipity(Jaccard)", "value": 4},
                        {"label": "Novelty", "value": 5},
                    ],
                ),
            ),
            dbc.Col([
                dbc.Input(id="rerank_alpha", placeholder="Type alpha value", type="float"),
                dbc.Button(id="rerank_reset", children="리랭킹 초기화"),
                dbc.Button(id="rerank_run", children="Rerank"),
            ]),
        ]
        return tmp

# 초기화 버튼 누를 때 선택 초기화
@callback(
    Output('rerank_obj', 'value'),
    Output('rerank_alpha', 'value'),
    Output('rerank_run', 'n_clicks'),
    Input('rerank_reset', 'n_clicks'),
)
def item_reset_selection(value):
    return [], [], 0


######################### 리랭킹 진행 ##############################
######################### 리랭킹 진행 ##############################
######################### 리랭킹 진행 ##############################

#백엔드에 리랭킹 계산 요청하고 받은 것으로 바로 모든 그림을 그려냄
#그리고 통째로 리턴
@callback(
    Output('user_deep_analysis', 'children'),
    Input('rerank_run', 'n_clicks'), #  선택하기 전까지 비활성화도 필요
    State('users_for_analysis', 'data'),
    State('rerank_obj', 'value'),
    State('rerank_alpha', 'value'),
    prevent_initial_call=True
)
def draw_rerank(value, user_lst, obj, alpha):
    if value != 1:
        raise PreventUpdate
    else:

        #todo: user_lst, obj, alpha를 통해 백엔드에 요청
        # 지표와 아이템들을 바로 merge할 수 있는 상태로 받는다.

        indicator = dbc.Row(
            children=[
                html.P('지표 비교. 리랭킹했더니 지표가 어떻게 변화했는지 +-로'),

            ],
        )
        item_poster = dbc.Row(
            children=[
                html.P('아이템 쪽에서의 포스터 처럼'),
                html.P('원래 많이 추천된 놈. 리랭킹했더니 많이 추천된 놈. 완전 뉴 페이스'),
            ],
        )
        item_side = dbc.Row(
            children=[

                html.P('리랭킹된 아이템들 사이드 정보. 상준이 장르. 년도까지'),
            ],
        )
        tmp = [indicator,item_poster,item_side]

        return tmp

#pd.DataFrame.from_dict(data=data, orient='tight')데이터프래임
# request.get().json
# 저장