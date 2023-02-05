import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate

# import feffery_antd_components as fac
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
def get_user_side_by_items(selected_item: list):
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


header_exp = html.Div(
    children=[
        dbc.Row(
            [
                html.Div("유저가 장바구니에 넣은 실험들"),
                dcc.Dropdown(
                    id="exp_id_for_deep_analysis",
                    options=["exp1"],
                ),
            ]
        )
    ]
)

header_user_or_item = html.Div(
    children=[
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

# test = html.Div(html.P(id="testin", children="gogo"))

layout = html.Div(
    children=[
        gct.get_navbar(has_sidebar=False),
        # test,
        html.Div(
            children=[
                header_exp,
                dbc.Spinner(header_user_or_item),
                # 스피너로 묶었는데 생각대로 안 나옴
                # dbc.Spinner(
                html.Div(
                    id="deep_analysis_page",
                ),
                # ),
            ],
            className="container",
            style={"margin-top": "4rem"} # navbar에 가려지는것 방지
        ),
        dcc.Store(id="trash"),  # 아무런 기능도 하지 않고, 그냥 콜백의 아웃풋 위치만 잡아주는 녀석
        dcc.Store(id="store_selected_exp"),
        dcc.Store(id="store_exp_names", storage_type="session"),
        dcc.Store(id="store_exp_ids", storage_type="session"),
        dcc.Store(id="store_exp_column", storage_type="session"),
    ],
)


# @callback(Output("testin", "children"), Input("store_selected_exp", "data"))
# def testing(val):
#     tmp = val
#     return str(tmp)


### 고객이 정한 장바구니가 담긴 store id가 필요함
##그거 토대로 버튼 그룹을 구성해야 함
@callback(
    Output("exp_id_for_deep_analysis", "options"),
    Input("store_exp_names", "data"),
    State("store_exp_ids", "data"),
)
def show_exp_choice(exp_name, exp_id):
    option = [{"label": i, "value": j} for i, j in zip(exp_name, exp_id)]
    return option


# 고객이 장바구니에서 원하는 exp_id를 선택하면 그에 대한 user, item을 불러온다.
@callback(
    Output("show_user_or_item", "options"),
    Output("show_user_or_item", "value"),
    Input("exp_id_for_deep_analysis", "value"),
    State("store_user_state", "data"),
    State("store_user_dataset", "data"),
    prevent_initial_call=True,
)
def choose_experiment(
    exp,
    vip,
    dataset_name,
):
    if exp is None:
        raise PreventUpdate
    global user
    global item
    global uniq_genre

    params = {"ID": vip["username"], "dataset_name": dataset_name, "exp_id": exp}
    # params = {"ID": 'mkdir', "dataset_name": 'ml-1m', "exp_id": 1}
    # print(params)
    user = requests.get(gct.API_URL + "/frontend/user_info", params=params).json()
    # print(user)
    user = pd.DataFrame.from_dict(data=user, orient="tight")
    # print(user)
    user.columns = [
        "user_id",
        "gender",
        "age",
        "occupation",
        "user_profile",
        "pred_item",
        "xs",
        "ys",
        "recall",
    ]

    user = user.set_index("user_id")
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
    item["recommended_users"] = item["recommended_users"].apply(
        lambda d: d if isinstance(d, list) else []
    )
    item["release_year"] = item["release_year"].astype(np.int16)
    item = item.set_index("item_id")
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
    # print(item.describe())
    return option, None


# 유저 페이지를 띄울지, 아이템 페이지를 띄울지
@callback(
    Output("deep_analysis_page", "children"),
    Input("show_user_or_item", "value"),
    Input("exp_id_for_deep_analysis", "value"),
    prevent_initial_call=True,
)
def display_overall(val, exp_id):
    if ctx.triggered_id == "exp_id_for_deep_analysis":
        return None
    else:
        if exp_id == None:
            return None
        if val == 1:  # 아이템을 선택함
            year_min = item["release_year"].min()
            year_max = item["release_year"].max()

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
                                            id="items_for_analysis",
                                            storage_type="session",
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
            option_age = html.Div(
                children=[
                    html.H3("옵션을 통한 선택"),
                    html.P("연령대"),
                    dcc.Checklist(
                        options=sorted(user["age"].unique()),
                        id="selected_age",
                    ),
                ]
            )
            option_gender = html.Div(
                children=[
                    html.P("성별"),
                    dcc.Dropdown(
                        options=["M", "F"],
                        id="selected_gender",
                    ),
                ]
            )
            option_occupation = html.Div(
                children=[
                    html.P("직업"),
                    dcc.Dropdown(
                        options=sorted(user["occupation"].unique()),
                        multi=True,
                        id="selected_occupation",
                    ),
                ]
            )
            option_wrong = html.Div(
                children=[
                    dcc.Checklist(
                        options=["틀린 유저들만(0.5 기준으로)"],
                        id="selected_wrong",
                    ),
                ]
            )

            user_selection = html.Div(
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    children=[
                                        option_age,
                                        option_gender,
                                        option_occupation,
                                        option_wrong,
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
                                            id="users_for_analysis",
                                            storage_type="session",
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

    selected_item = item.loc[item["selected"] == "Selected"]
    Notselected_item = item.loc[item["selected"] != "Selected"]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=Notselected_item["xs"],
            y=Notselected_item["ys"],
            name="Not selected",
            marker_color="red",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=selected_item["xs"],
            y=selected_item["ys"],
            name="selected       ",
            mode="markers",
            marker_color="green",
        )
    )
    fig.add_trace(go.Scatter(x=[0], y=[0], name=" ", marker_color="white"))

    fig.update_traces(mode="markers", opacity=0.6)
    fig.update_layout(
        title="Item embedding plot",
        yaxis_zeroline=True,
        xaxis_zeroline=False,
        margin={},
    )
    return fig


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
        user_lst = user_lst[user_lst["recall"] <= 0.5]
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
    Output("user_emb_graph", "figure"),
    Input("users_selected_by_option", "data"),
)
def update_graph(store1):
    user["selected"] = "Not Selected"
    user.loc[store1, "selected"] = "Selected"

    selected_user = user.loc[user["selected"] == "Selected"]
    Notselected_user = user.loc[user["selected"] != "Selected"]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=Notselected_user["xs"],
            y=Notselected_user["ys"],
            name="Not selected",
            marker_color="red",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=selected_user["xs"],
            y=selected_user["ys"],
            name="selected       ",
            mode="markers",
            marker_color="green",
        )
    )
    fig.add_trace(go.Scatter(x=[0], y=[0], name=" ", marker_color="white"))

    fig.update_traces(mode="markers", opacity=0.6)
    fig.update_layout(
        title="user embedding plot",
        yaxis_zeroline=True,
        xaxis_zeroline=False,
        margin={},
    )
    return fig
    emb = px.scatter(
        user,
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
    State("exp_id_for_deep_analysis", "value"),
    State("store_user_state", "data"),
    State("store_user_dataset", "data"),
    prevent_initial_call=True,
)
def draw_rerank(value, user_lst, obj, alpha, exp_id, id, dataset):
    if value != 1:
        raise PreventUpdate
    else:
        tmp = user.loc[user_lst]
        # TODO: user_lst, obj, alpha를 통해 백엔드에 리랭킹 요청
        params = {
            "user_lst": user_lst,
            "obj": obj,
            "alpha": alpha,
            "ID": id,
            "dataset": dataset,
            "exp_id": exp_id,
        }
        diff_metric, reranked_lst = requests.get(
            gct.API_URL + "/rerank_selected_users", params
        )
        # diff는 데이터프레임화 가능
        # reranked는 시리즈화 가능
        tmp["reranked_item"] = reranked_lst
        # 지표와 아이템들을 바로 merge할 수 있는 상태로 받는다.

        # 첫번째 - 지표 변화
        sub = diff_metric.loc["rerank"] - diff_metric.loc["origin"]

        # 두번째 - 아이템 포스터
        origin_item = set()
        rerank_item = set()
        profile_item = set()
        for i in user_lst:
            origin_item |= set(i["pred_item"])
            rerank_item |= set(i["reranked_item"])
            profile_item |= set(i["user_profile"])
        new_item = rerank_item - origin_item
        pop = (
            item.loc[list(origin_item)]
            .sort_values(by=["len"], ascending=False)
            .head(10)
            .index
        )
        pop_lst = [make_card(item) for item in pop]
        rer = (
            item.loc[list(rerank_item)]
            .sort_values(by=["len"], ascending=False)
            .head(10)
            .index
        )
        rer_lst = [make_card(item) for item in rer]
        new = (
            item.loc[list(new_item)]
            .sort_values(by=["len"], ascending=False)
            .head(10)
            .index
        )
        new_lst = [make_card(item) for item in new]

        # 세번째 - 사이드 정보
        def plot_usergroup_genre(user_lst):
            """
            need variance :
            dataset = dataset.info class
            run1 = model_managers['EASE'].get_all_model_configs()[0] 과 같은 실험 정보
            user_df = pd.merge(dataset.user_df, dataset.ground_truth, on='user_id')
            중복 아이템을 셀지 말지의 여부를 가르는 문제가 있따.
            """

            # ['user_id', 'gender', 'age', 'occupation', 'user_profile', 'pred_item', 'xs', 'ys', 'recall']
            ### 함수화 하면 좋을 부분 ###
            total_item_genre = Counter()
            user_profile = Counter()
            user_rec = Counter()
            user_rerank = Counter()

            for i in item["genre"]:
                total_item_genre += Counter(i.split())
            for i in origin_item:
                user_rec += Counter(item.loc[i]["genre"].split())
            for i in rerank_item:
                user_rerank += Counter(item.loc[i]["genre"].split())
            for i in profile_item:
                user_profile += Counter(item.loc[i]["genre"].split())

            user_profile = dict(
                sorted(user_profile.items(), key=lambda x: x[1], reverse=True)
            )
            user_rec = dict(sorted(user_rec.items(), key=lambda x: x[1], reverse=True))
            total_item_genre = dict(
                sorted(total_item_genre.items(), key=lambda x: x[1], reverse=True)
            )
            user_rerank = dict(
                sorted(user_rerank.items(), key=lambda x: x[1], reverse=True)
            )

            user_profile_labels = list(user_profile.keys())
            user_profile_values = list(user_profile.values())
            user_rec_labels = list(user_rec.keys())
            user_rec_values = list(user_rec.values())
            user_rerank_labels = list(user_rerank.keys())
            user_rerank_values = list(user_rerank.values())
            total_item_genre_labels = list(total_item_genre.keys())
            total_item_genre_values = list(total_item_genre.values())

            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[
                    [{"type": "domain"}, {"type": "domain"}],
                    [{"type": "domain"}, {"type": "domain"}],
                ],
                subplot_titles=(
                    "total item",
                    "profile",
                    "reccomend list",
                    "rerank list",
                ),
            )
            fig.add_trace(
                go.Pie(
                    labels=total_item_genre_labels,
                    values=total_item_genre_values,
                    name="total item genre",
                    pull=[0.07] + [0] * (len(total_item_genre_values) - 1),
                ),  # textinfo='label+percent', pull=[0.2]+[0]*(len(total_item_genre_values)-1)
                1,
                1,
            )
            fig.add_trace(
                go.Pie(
                    labels=user_profile_labels,
                    values=user_profile_values,
                    name="user profile genre",
                    pull=[0.07] + [0] * (len(user_profile_values) - 1),
                ),  # textinfo='label+percent', pull=[0.2]+[0]*(len(user_profile_values)-1)
                1,
                2,
            )
            fig.add_trace(
                go.Pie(
                    labels=user_rec_labels,
                    values=user_rec_values,
                    name="user Rec list genre",
                    pull=[0.07] + [0] * (len(user_rec_values) - 1),
                ),  # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rec_values)-1)
                2,
                1,
            )
            fig.add_trace(
                go.Pie(
                    labels=user_rerank_labels,
                    values=user_rerank_values,
                    name="user rerank",
                    pull=[0.07] + [0] * (len(user_rerank_values) - 1),
                ),  # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rerank_values)-1)
                2,
                2,
            )

            fig.update_traces(hole=0.3, hoverinfo="label+percent+name")

            fig.add_annotation(
                text=f"Total users num in this group : ",
                x=0.5,
                y=0.5,
                font_size=20,
                showarrow=False,
            )
            fig.update_layout(
                title_text=f"age:, gender:, occupation: User group genre pie chart",
                width=1000,
                height=800,
            )

            return fig

        # 유저별 추천 아이템(10개), 리랭크 아이템 10개 필요하다. item poster위해
        #

        indicator = dbc.Row(
            children=[
                html.P("지표 비교. 리랭킹했더니 지표가 어떻게 변화했는지 +-로"),
                # TODO: 기존 지표와 리랭킹 지표 차이를 각각 표시.
                [
                    dbc.Badge(children=i, color="primary" if i < 0 else "danger")
                    for metric, i in zip(sub.index, sub)
                ],
            ],
        )
        item_poster = html.Div(
            children=[
                html.H3("기존 많이 추천된 아이템 top 10"),
                dbc.Row(children=pop_lst, style={"overflow": "scroll", "height": 500}),
                html.H3("리랭킹 후 많이 추천된 아이템 top 10"),
                dbc.Row(children=rer_lst, style={"overflow": "scroll", "height": 500}),
                html.H3("기존에 추천되지 않은 아이템 top 10"),
                dbc.Row(children=new_lst, style={"overflow": "scroll", "height": 500}),
            ],
        )
        item_side = dbc.Row(
            children=[
                html.P("리랭킹된 아이템들 사이드 정보. 상준이 장르. 년도까지"),
                # 장르 파이차트에서 유저 프로파일 필요. 이외에는 사용되지 않음
                dcc.Graph(figure=plot_usergroup_genre(tmp)),
            ],
        )
        children = [indicator, item_poster, item_side]

        return children
