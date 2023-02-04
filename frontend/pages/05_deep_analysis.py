import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
import feffery_antd_components as fac
from . import global_component as gct
from collections import Counter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import requests

dash.register_page(__name__, path="/deep_analysis")


user = None
item = None
uniq_genre = None

# load_figure_template("darkly") # figure 스타일 변경

# 포스터를 띄우는 함수
def make_card(element):
    tmp = item.loc[element]
    card = dbc.Col(
        children=dbc.Card(
            [
                dbc.CardImg(top=True),  # 여기에 이미지를 넣으면 된다.
                dbc.CardBody(
                    [
                        html.H6(tmp["movie_title"]),
                        html.P(tmp["genre"]),
                        html.P(tmp["release_year"]),
                        html.P(tmp["item_pop"]),
                    ],
                ),
            ],
        ),
        width={"size": 3},
    )
    return card


# related user 그래프 그리는 준비하는 함수
def get_user_side_by_items(selected_item: list) -> tuple:
    """
    선택된 item들의 idx를 넣어주면, 그 아이템들을 사용한 유저, 추천받은 유저들의 인구통계학적 정보 수집
    총 6개의 Counter가 return, 앞에서 부터 2개씩 age, gender, occupation 정보
    e.g., 앞의 age는 사용한 유저, 뒤의 age는 추천받은 유저들 ...
    """
    # Counter 세팅
    age_Counter_profile, gender_Counter_profile, occupation_Counter_profile = (
        Counter(),
        Counter(),
        Counter(),
    )
    age_Counter_rec, gender_Counter_rec, occupation_Counter_rec = (
        Counter(),
        Counter(),
        Counter(),
    )

    for idx in selected_item:
        one_item = item.loc[idx]

        # profile Counter
        tmp = user.loc[one_item["item_profile_user"], ["age", "gender", "occupation"]]
        age_Counter_profile += Counter(tmp["age"])
        gender_Counter_profile += Counter(tmp["gender"])
        occupation_Counter_profile += Counter(tmp["occupation"])

        # profile Counter
        if one_item.isnull()["recommended_users"]:
            continue
        tmp = user.loc[one_item["recommended_users"], ["age", "gender", "occupation"]]
        age_Counter_rec += Counter(tmp["age"])
        gender_Counter_rec += Counter(tmp["gender"])
        occupation_Counter_rec += Counter(tmp["occupation"])

    age_Counter_profile = dict(
        sorted(age_Counter_profile.items(), key=lambda x: x[1], reverse=True)
    )
    age_Counter_rec = dict(
        sorted(age_Counter_rec.items(), key=lambda x: x[1], reverse=True)
    )

    gender_Counter_profile = dict(
        sorted(gender_Counter_profile.items(), key=lambda x: x[1], reverse=True)
    )
    gender_Counter_rec = dict(
        sorted(gender_Counter_rec.items(), key=lambda x: x[1], reverse=True)
    )

    occupation_Counter_profile = dict(
        sorted(occupation_Counter_profile.items(), key=lambda x: x[1], reverse=True)
    )
    occupation_Counter_rec = dict(
        sorted(occupation_Counter_rec.items(), key=lambda x: x[1], reverse=True)
    )

    return (
        age_Counter_profile,
        age_Counter_rec,
        gender_Counter_profile,
        gender_Counter_rec,
        occupation_Counter_profile,
        occupation_Counter_rec,
    )


def plot_age_counter(age_Counter_profile: Counter, age_Counter_rec: Counter):
    age_Counter_profile_labels = list(age_Counter_profile.keys())
    age_Counter_profile_values = list(age_Counter_profile.values())
    age_Counter_rec_labels = list(age_Counter_rec.keys())
    age_Counter_rec_values = list(age_Counter_rec.values())
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("Age rtio(profile)", "Age ratio(rec)"),
    )
    fig.add_trace(
        go.Pie(
            labels=age_Counter_profile_labels,
            values=age_Counter_profile_values,
            name="Age(profile)",
            pull=[0.07] + [0] * (len(age_Counter_profile_values) - 1),
        ),  # textinfo='label+percent', pull=[0.2]+[0]*(len(total_item_genre_values)-1)
        1,
        1,
    )
    fig.add_trace(
        go.Pie(
            labels=age_Counter_rec_labels,
            values=age_Counter_rec_values,
            name="Age(rec)",
            pull=[0.07] + [0] * (len(age_Counter_rec_values) - 1),
        ),  # textinfo='label+percent', pull=[0.2]+[0]*(len(user_profile_values)-1)
        1,
        2,
    )

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
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("Gender ratio(profile)", "Gender ratio(rec)"),
    )
    fig.add_trace(
        go.Pie(
            labels=gender_Counter_profile_labels,
            values=gender_Counter_profile_values,
            name="user Rec list genre",
            pull=[0.07] + [0] * (len(gender_Counter_profile_values) - 1),
        ),
        1,
        1,
    )
    fig.add_trace(
        go.Pie(
            labels=gender_Counter_rec_labels,
            values=gender_Counter_rec_values,
            name="user rerank",
            pull=[0.07] + [0] * (len(gender_Counter_rec_values) - 1),
        ),
        1,
        2,
    )
    fig.update_traces(hole=0.3, hoverinfo="label+percent+name")
    fig.update_layout(
        title_text="Selected Item profile vs Selected Item rec list (Gender)",
        # width=1000,
        # height=500
    )

    return fig


def plot_occupation_counter(
    occupation_Counter_profile: Counter, occupation_Counter_rec: Counter
):
    occupation_Counter_profile_labels = list(occupation_Counter_profile.keys())
    occupation_Counter_profile_values = list(occupation_Counter_profile.values())
    occupation_Counter_rec_labels = list(occupation_Counter_rec.keys())
    occupation_Counter_rec_values = list(occupation_Counter_rec.values())
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("Occupation ratio(profile)", "Occupation ratio(rec)"),
    )
    fig.add_trace(
        go.Pie(
            labels=occupation_Counter_profile_labels,
            values=occupation_Counter_profile_values,
            name="user Rec list genre",
            pull=[0.07] + [0] * (len(occupation_Counter_profile_values) - 1),
        ),  # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rec_values)-1)
        1,
        1,
    )
    fig.add_trace(
        go.Pie(
            labels=occupation_Counter_rec_labels,
            values=occupation_Counter_rec_values,
            name="user rerank",
            pull=[0.07] + [0] * (len(occupation_Counter_rec_values) - 1),
        ),  # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rerank_values)-1)
        1,
        2,
    )
    fig.update_traces(hole=0.3, hoverinfo="label+percent+name")
    fig.update_layout(
        title_text="Selected Item profile vs Selected Item rec list (Occupation)",
        # width=1000,
        # height=500
    )

    return fig


header = html.Div(
    children=[
        dbc.Row(
            [
                html.Div("유저가 장바구니에 넣은 실험들"),
                dcc.Dropdown(
                    id="exp_id_for_deep_analysis",
                    options=["exp1"],
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div("해당 실험의 아이템, 유저 페이지"),
                dbc.RadioItems(
                    id="show_user_or_item",
                    className="btn-group",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-primary",
                    labelCheckedClassName="active",
                ),
            ]
        ),
    ]
)


item_top = html.Div(
    id="item_top_poster",
    children=[html.P("골라야 볼 수 있습니다.")],
)

item_related_users = html.Div(
    id="item_related_users",
)

# 리랭크 박스
user_rerank = dbc.Row(id="rerank_box")

# 유저 분석
user_analysis = html.Div(id="user_deep_analysis")

test = html.Div(html.P(id="testin", children="gogo"))

layout = html.Div(
    children=[
        gct.get_navbar(has_sidebar=False),
        test,
        html.Div(
            children=[
                header,
                html.Div(
                    id="deep_analysis_page",
                ),
            ],
            className="container",
        ),
        dcc.Store(id="trash"),  # 아무런 기능도 하지 않고, 그냥 콜백의 아웃풋 위치만 잡아주는 녀석
        dcc.Store(id="store_selected_exp"),
        dcc.Store(id='store_exp_names', storage_type='session'),
        dcc.Store(id='store_exp_ids', storage_type='session'),
        dcc.Store(id='store_exp_column', storage_type='session')
    ],
)


@callback(Output("testin", "children"), Input("store_selected_exp", "data"))
def testing(val):
    tmp = val
    return str(tmp)


### 고객이 정한 장바구니가 담긴 store id가 필요함
##그거 토대로 버튼 그룹을 구성해야 함
@callback(
    Output('exp_id_for_deep_analysis', 'options'),
    Input('store_exp_names', 'data'),
    State('store_exp_ids', 'data'),
)
def show_exp_choice(exp_name, exp_id):
    option = [{'label': i, 'value': j} for i,j in zip(exp_name, exp_id)]
    return option

# 고객이 장바구니에서 원하는 exp_id를 선택하면 그에 대한 user, item을 불러온다.
@callback(
    Output("show_user_or_item", "options"),
    Input("exp_id_for_deep_analysis", "value"),
    State("store_user_state", "data"),
    State("store_user_dataset", "data"),
    prevent_initial_call=True,
)
def choose_experiment(exp, vip, dataset_name, ):
    if exp is None:
        raise PreventUpdate
    global user
    global item
    global uniq_genre

    params = {"ID": vip['username'], "dataset_name": dataset_name, "exp_id": exp}
    user = requests.get(gct.API_URL + "/frontend/user_info", params=params).json()
    user = pd.DataFrame.from_dict(data=user, orient="tight")
    user.columns = ['user_id', 'gender', 'age', 'occupation', 'user_profile', 'pred_item', 'xs', 'ys']
    user = user.set_index('user_id')
    item = requests.get(gct.API_URL + "/frontend/item_info", params=params).json()
    item = pd.DataFrame.from_dict(data=item, orient="tight")
    item.columns = [
        "item_id",
        "movie_title",
        "genre",
        "release_year",
        "item_pop",
        "item_profile_user",
        "recommended_users",
        "xs",
        "ys",
    ]
    item["release_year"] = item["release_year"].astype(np.int16)
    item = item.set_index('item_id')
    item["selected"] = 0
    item["len"] = item["recommended_users"].apply(len)
    uniq_genre = set()
    for i in item["genre"]:
        uniq_genre |= set(i.split(" "))
    uniq_genre = [*uniq_genre]

    option = [
        {"label": "item", "value": 1},
        {"label": "user", "value": 2},
    ]
    return option


# 유저 페이지를 띄울지, 아이템 페이지를 띄울지
@callback(
    Output("deep_analysis_page", "children"),
    Input("show_user_or_item", "value"),
    prevent_initial_call=True,
)
def display_overall(val):
    if val == 1:  # 아이템을 선택함
        year_min = item['release_year'].min()
        year_max = item['release_year'].max()

        item_selection = html.Div(
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                children=[
                                    html.H3("옵션을 통한 선택"),
                                    html.P("장르"),
                                    dcc.Dropdown(
                                        options=uniq_genre,
                                        multi=True,
                                        id="selected_genre",
                                    ),
                                    html.P("년도"),
                                    dcc.RangeSlider(
                                        min=year_min,
                                        max=year_max,
                                        value=[
                                            year_min,
                                            year_max,
                                        ],
                                        step=1,
                                        marks=None,
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                        allowCross=False,
                                        id="selected_year",
                                    ),
                                    html.P("인기도"),
                                    dbc.Button(
                                        id="item_reset_selection",
                                        children="초기화",
                                        color="primary",
                                    ),
                                    html.P(id="n_items"),
                                    dbc.Button(id="item_run", children="RUN"),
                                    dcc.Store(
                                        id="items_selected_by_option",
                                        storage_type="session",
                                    ),  # 데이터를 저장하는 부분
                                    dcc.Store(
                                        id="items_selected_by_embed",
                                        storage_type="session",
                                    ),  # 데이터를 저장하는 부분
                                    dcc.Store(
                                        id="items_for_analysis", storage_type="session"
                                    ),  # 데이터를 저장하는 부분
                                ],
                                # className='form-style'
                            ),
                            width=3,
                        ),
                        dbc.Col(
                            html.Div(
                                children=[
                                    html.H3("아이템 2차원 임베딩"),
                                    html.P("참고로 리랭킹 관련한 지원은 유저 페이지에서만 됩니다."),
                                    html.Br(),
                                    dcc.Graph(
                                        id="item_emb_graph",
                                    ),
                                ],
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            html.Div(
                                children=[
                                    html.H3("사이드인포"),
                                    html.Div(id="item_side_graph"),
                                ],
                                style={"overflow": "scroll", "height": 700},
                            ),
                            width=3,
                        ),
                    ],
                )
            ]
        )
        return [item_selection, item_top, item_related_users]
        # 얘네는 run 버튼 이후로 되게 하자
    else:
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
                                        options=["틀린 유저들만(0.5 기준으로)"],
                                        value=[],
                                        id="selected_wrong",
                                    ),
                                    html.Br(),
                                    dbc.Button(
                                        id="user_reset_selection",
                                        children="초기화",
                                        color="primary",
                                    ),
                                    html.P(id="n_users"),
                                    dbc.Button(id="user_run", children="RUN"),
                                    dcc.Store(
                                        id="users_selected_by_option",
                                        storage_type="session",
                                    ),  # 데이터를 저장하는 부분
                                    dcc.Store(
                                        id="users_selected_by_embed",
                                        storage_type="session",
                                    ),  # 데이터를 저장하는 부분
                                    dcc.Store(
                                        id="users_for_analysis", storage_type="session"
                                    ),  # 데이터를 저장하는 부분
                                ],
                                # className='form-style'
                            ),
                            width=3,
                        ),
                        dbc.Col(
                            html.Div(
                                children=[
                                    html.H3("유저 2차원 임베딩"),
                                    dcc.Graph(
                                        id="user_emb_graph",
                                        style={"config.responsive": True},
                                    ),
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
                                style={"overflow": "scroll", "height": 700},
                            ),
                            width=3,
                        ),
                    ],
                ),
            ]
        )
        return [
            user_selection,
            user_rerank,
            user_analysis,
        ]


#########################################################
######################## 아이템 ##########################
#########################################################

# 옵션으로 선택한 아이템을 store1에 저장
@callback(
    Output("items_selected_by_option", "data"),
    Input("selected_genre", "value"),
    Input("selected_year", "value"),
)
def save_items_selected_by_option(genre, year):
    item_lst = item.copy()
    item_lst = item_lst[
        item_lst["genre"].str.contains(
            "".join([*map(lambda x: f"(?=.*{x})", genre)]) + ".*", regex=True
        )
    ]
    item_lst = item_lst[
        (item_lst["release_year"] >= year[0]) & (item_lst["release_year"] <= year[1])
    ]
    return item_lst.index.to_list()


# embed graph에서 선택한 아이템을 store2에 저장
@callback(
    Output("items_selected_by_embed", "data"), Input("item_emb_graph", "selectedData")
)
def save_items_selected_by_embed(emb):
    if emb is None:
        raise PreventUpdate
    item_idx = [i["pointNumber"] for i in emb["points"]]
    item_lst = item.iloc[item_idx]
    return item_lst.index.to_list()


# 최근에 선택한 아이템을 최종 store에 저장
@callback(
    Output("items_for_analysis", "data"),
    Output("n_items", "children"),
    Input("items_selected_by_option", "data"),
    Input("items_selected_by_embed", "data"),
)
def prepare_analysis(val1, val2):
    if ctx.triggered_id == "items_selected_by_option":
        return val1, f"selected items: {len(val1)}"
    else:
        return val2, f"selected items: {len(val2)}"


# 최근에 저장된 store 기준으로 임베딩 그래프를 그림
@callback(
    Output("item_emb_graph", "figure"),
    Input("items_selected_by_option", "data"),
)
def update_graph(store1):
    item["selected"] = "Not Selected"
    item.loc[store1, "selected"] = "Selected"
    emb = px.scatter(
        item,
        x="xs",
        y="ys",
        color="selected",  # 갯수에 따라 색깔이 유동적인 것 같다..
        opacity=0.9,
        marginal_x="histogram",
        marginal_y="histogram",
    )
    emb.update_layout(
        clickmode="event+select",
        width=700,
        height=700,
    )
    return emb


# 최근에 저장된 store 기준으로 사이드 그래프를 그림
@callback(
    Output("item_side_graph", "children"),
    Input("items_selected_by_option", "data"),
    Input("items_selected_by_embed", "data"),
)
def update_graph(store1, store2):
    if ctx.triggered_id == "items_selected_by_option":
        tmp = item.loc[store1]
        year = px.histogram(tmp, x="release_year")
        genre = px.histogram(tmp, x="genre")
        return (dcc.Graph(figure=year), dcc.Graph(figure=genre))
    else:
        if not store2:
            raise PreventUpdate
        tmp = item.loc[store2]
        tmp = tmp[tmp["selected"] == "Selected"]
        year = px.histogram(tmp, x="release_year")
        genre = px.histogram(tmp, x="genre")
        return (dcc.Graph(figure=year), dcc.Graph(figure=genre))


# 초기화 버튼 누를 때 선택 초기화
@callback(
    Output("selected_genre", "value"),
    Output("selected_year", "value"),
    Output("item_run", "n_clicks"),
    Input("item_reset_selection", "n_clicks"),
)
def item_reset_selection(value):
    return [], [item["release_year"].min(), item["release_year"].max()], 0


#### item run 실행 시 실행될 함수들 #####

# 아이템 포스터 카드 표시(top pop 10, top rec 10)
@callback(
    Output("item_top_poster", "children"),
    Input("item_run", "n_clicks"),
    State("items_for_analysis", "data"),
    prevent_initial_call=True,
)
def draw_item_top(value, data):
    if value != 1:
        raise PreventUpdate
    else:
        pop = (
            item.loc[data].sort_values(by=["item_pop"], ascending=False).head(10).index
        )
        pop_lst = [make_card(item) for item in pop]  # 보여줄 카드 갯수 지정 가능
        rec = item.loc[data].sort_values(by=["len"], ascending=False).head(10).index
        rec_lst = [make_card(item) for item in rec]  # 보여줄 카드 갯수 지정 가능
        children = [
            html.H3("선택한 아이템 인기도 top 10"),
            dbc.Row(children=pop_lst, style={"overflow": "scroll", "height": 500}),
            html.H3("선택한 아이템 추천횟수 top 10"),
            dbc.Row(children=rec_lst, style={"overflow": "scroll", "height": 500}),
            html.Br(),
        ]
        return children


# 아이템 관련 유저 그리기
@callback(
    Output("item_related_users", "children"),
    Input("item_run", "n_clicks"),
    State("items_for_analysis", "data"),
    prevent_initial_call=True,
)
def draw_item_related_users(value, data):
    if value != 1:
        raise PreventUpdate
    else:
        (
            age_Counter_profile,
            age_Counter_rec,
            gender_Counter_profile,
            gender_Counter_rec,
            occupation_Counter_profile,
            occupation_Counter_rec,
        ) = get_user_side_by_items(data)
        age = dcc.Graph(figure=plot_age_counter(age_Counter_profile, age_Counter_rec))
        gender = dcc.Graph(
            figure=plot_gender_counter(gender_Counter_profile, gender_Counter_rec)
        )
        occupation = dcc.Graph(
            figure=plot_occupation_counter(
                occupation_Counter_profile, occupation_Counter_rec
            )
        )

        children = [
            html.H3("유저 프로필, 유저 추천 리스트"),
            dbc.Row(
                [
                    dbc.Col(age),
                    dbc.Col(gender),
                    dbc.Col(occupation),
                ]
            ),
        ]
        return children


#########################################################
######################## 유저 ###########################
#########################################################


# 선택한 유저들을 store1에 저장
@callback(
    Output("users_selected_by_option", "data"),
    Input("selected_age", "value"),
    Input("selected_gender", "value"),
    Input("selected_occupation", "value"),
    Input("selected_wrong", "value"),
)
def save_users_selected_by_option(age, gender, occupation, wrong):
    user_lst = user.copy()
    if age:
        user_lst = user_lst[user_lst["age"].isin(age)]
    if gender:
        user_lst = user_lst[user_lst["gender"] == gender]
    if occupation:
        user_lst = user_lst[user_lst["occupation"].isin(occupation)]
    if wrong:
        user_lst = user_lst[user_lst["div_jac"] <= 0.77]  # 당장의 데이터에 recall이 없다.
    return user_lst.index.to_list()


# user embed graph에서 선택한 유저들을 store2에 저장
@callback(
    Output("users_selected_by_embed", "data"), Input("user_emb_graph", "selectedData")
)
def save_users_selected_by_embed(emb):
    if emb is None:
        raise PreventUpdate
    user_idx = [i["pointNumber"] for i in emb["points"]]
    user_lst = user.iloc[user_idx]
    return user_lst.index.to_list()


# 최근에 선택한 유저를 최종 store에 저장
@callback(
    Output("users_for_analysis", "data"),
    Output("n_users", "children"),
    Input("users_selected_by_option", "data"),
    Input("users_selected_by_embed", "data"),
)
def prepare_analysis(val1, val2):
    if ctx.triggered_id == "users_selected_by_option":
        return val1, f"selected users: {len(val1)}"
    else:
        return val2, f"selected users: {len(val2)}"


# 최근에 저장된 store 기준으로 유저 임베딩 그래프를 그림
@callback(
    Output('user_emb_graph', 'figure'),
    Input('users_selected_by_option', 'data'),
)
def update_graph(store1):
    user['selected'] = 'Not Selected'
    user.loc[store1, 'selected'] = 'Selected'
    emb = px.scatter(
        user, x = 'xs', y = 'ys', color='selected', # 갯수에 따라 색깔이 유동적인 것 같다..
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

# #최근에 저장된 store 기준으로 사이드 그래프를 그림
@callback(
    Output("user_side_graph", "children"),
    Input("users_selected_by_option", "data"),
    Input("users_selected_by_embed", "data"),
)
def update_graph(store1, store2):
    def plot_age_counter(age_Counter_profile: Counter):
        age_Counter_profile_labels = list(age_Counter_profile.keys())
        age_Counter_profile_values = list(age_Counter_profile.values())
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "domain"}]],
            subplot_titles=("Age ratio(profile)"),
        )
        fig.add_trace(
            go.Pie(
                labels=age_Counter_profile_labels,
                values=age_Counter_profile_values,
                name="Age(profile)",
                pull=[0.07] + [0] * (len(age_Counter_profile_values) - 1),
            ),  # textinfo='label+percent', pull=[0.2]+[0]*(len(total_item_genre_values)-1)
            1,
            1,
        )

        fig.update_traces(hole=0.3, hoverinfo="label+percent+name")
        return fig

    def plot_gender_counter(gender_Counter_profile: Counter):
        gender_Counter_profile_labels = list(gender_Counter_profile.keys())
        gender_Counter_profile_values = list(gender_Counter_profile.values())

        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "domain"}]],
            subplot_titles=("Gender ratio(profile)"),
        )
        fig.add_trace(
            go.Pie(
                labels=gender_Counter_profile_labels,
                values=gender_Counter_profile_values,
                name="user Rec list genre",
                pull=[0.07] + [0] * (len(gender_Counter_profile_values) - 1),
            ),  # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rec_values)-1)
            1,
            1,
        )
        fig.update_traces(hole=0.3, hoverinfo="label+percent+name")

        return fig

    def plot_occupation_counter(occupation_Counter_profile: Counter):
        occupation_Counter_profile_labels = list(occupation_Counter_profile.keys())
        occupation_Counter_profile_values = list(occupation_Counter_profile.values())
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "domain"}]],
            subplot_titles=("Occupation ratio(profile)"),
        )
        fig.add_trace(
            go.Pie(
                labels=occupation_Counter_profile_labels,
                values=occupation_Counter_profile_values,
                name="user Rec list genre",
                pull=[0.07] + [0] * (len(occupation_Counter_profile_values) - 1),
            ),  # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rec_values)-1)
            1,
            1,
        )
        fig.update_traces(hole=0.3, hoverinfo="label+percent+name")
        return fig

    if ctx.triggered_id == "users_selected_by_option":
        tmp = user.loc[store1]
        age = plot_age_counter(Counter(tmp["age"]))
        gender = plot_gender_counter(Counter(tmp["gender"]))
        occupation = plot_occupation_counter(Counter(tmp["occupation"]))
        return (
            dcc.Graph(figure=age),
            dcc.Graph(figure=gender),
            dcc.Graph(figure=occupation),
        )
    else:
        if not store2:
            raise PreventUpdate
        tmp = user.loc[store2]
        tmp = tmp[tmp["selected"] == "Selected"]
        age = plot_age_counter(Counter(tmp["age"]))
        gender = plot_gender_counter(Counter(tmp["gender"]))
        occupation = plot_occupation_counter(Counter(tmp["occupation"]))
        return (
            dcc.Graph(figure=age),
            dcc.Graph(figure=gender),
            dcc.Graph(figure=occupation),
        )


# 초기화 버튼 누를 때 선택 초기화
@callback(
    Output("selected_age", "value"),
    Output("selected_gender", "value"),
    Output("selected_occupation", "value"),
    Output("selected_wrong", "value"),
    Output("user_run", "n_clicks"),
    Input("user_reset_selection", "n_clicks"),
)
def item_reset_selection(value):
    return [], [], [], [], 0


#### run 실행 시 실행될 함수들 #####

# 최종 store가 업데이트됐을 때 리랭킹 선택지 등장
@callback(
    Output("rerank_box", "children"),
    Input("user_run", "n_clicks"),
    prevent_initial_call=True,
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
            dbc.Col(
                [
                    dbc.Input(
                        id="rerank_alpha",
                        placeholder="Type alpha value",
                        # type="float",
                        min=0,
                        max=1,
                        step=0.1,
                        value=0.5,
                    ),
                    dbc.Button(id="rerank_reset", children="리랭킹 초기화"),
                    dbc.Button(id="rerank_run", children="Rerank"),
                ]
            ),
        ]
        return tmp


# 초기화 버튼 누를 때 선택 초기화
@callback(
    Output("rerank_obj", "value"),
    Output("rerank_alpha", "value"),
    Output("rerank_run", "n_clicks"),
    Input("rerank_reset", "n_clicks"),
)
def item_reset_selection(value):
    return [], [], 0


######################### 리랭킹 진행 ##############################
######################### 리랭킹 진행 ##############################
######################### 리랭킹 진행 ##############################

# 백엔드에 리랭킹 계산 요청하고 받은 것으로 바로 모든 그림을 그려냄
# 그리고 통째로 리턴
@callback(
    Output("user_deep_analysis", "children"),
    Input("rerank_run", "n_clicks"),  #  선택하기 전까지 비활성화도 필요
    State("users_for_analysis", "data"),
    State("rerank_obj", "value"),
    State("rerank_alpha", "value"),
    prevent_initial_call=True,
)
def draw_rerank(value, user_lst, obj, alpha):
    if value != 1:
        raise PreventUpdate
    else:
        origin_lst = user.loc[user_lst]
        # TODO: user_lst, obj, alpha를 통해 백엔드에 리랭킹 요청
        # 지표와 아이템들을 바로 merge할 수 있는 상태로 받는다.

        # reranked = requests.get(das;ldfj)
        # lst = origin_lst.merge(reranked)
        # 유저별 모든 지표가 필요하다. indicator때 비교하기 위해
        # 유저별 추천 아이템(10개), 리랭크 아이템 10개 필요하다. item poster위해
        #

        indicator = dbc.Row(
            children=[
                html.P("지표 비교. 리랭킹했더니 지표가 어떻게 변화했는지 +-로"),
                # TODO: 기존 지표와 리랭킹 지표 차이를 각각 표시.
                #
            ],
        )
        item_poster = dbc.Row(
            children=[
                html.P("아이템 쪽에서의 포스터 처럼"),
                html.P("원래 많이 추천된 놈. 리랭킹했더니 많이 추천된 놈. 완전 뉴 페이스"),
                # TODO 아
            ],
        )
        item_side = dbc.Row(
            children=[
                html.P("리랭킹된 아이템들 사이드 정보. 상준이 장르. 년도까지"),
                # 장르 파이차트에서 유저 프로파일 필요. 이외에는 사용되지 않음
            ],
        )
        tmp = [indicator, item_poster, item_side]

        return tmp
