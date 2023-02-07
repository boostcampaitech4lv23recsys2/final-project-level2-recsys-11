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

# def make_card(element):
#     tmp = item.loc[element]
#     img = 'https://s3.us-west-2.amazonaws.com/secure.notion-static.com/3e0c51b2-0058-4c7e-a081-63c36afbb9ab/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230206%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230206T092314Z&X-Amz-Expires=86400&X-Amz-Signature=459ae00b0cb7fe0924f62f17992549f0cb0de1fd2db35bd510675575b2c2ba8e&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject' if tmp['item_url'] == '' else tmp['item_url']

#     card = dbc.Col(
#         children=dbc.Card(
#             [
#                 dbc.CardImg(src=img, top=True),
#                 dbc.CardBody(
#                     [
#                         html.H6(tmp["movie_title"]),
#                         html.P(tmp["genre"]),
#                         html.P(f'출시년도 {tmp["release_year"]}'),
#                         html.P(f'인기도 {round(tmp["item_pop"] * 100, 3)}%'),
#                     ],
#                 ),
#             ],
#         ),
#         width={"size": 3},
#     )
#     return card



# related user 그래프 그리는 준비하는 함수
def get_user_side_by_items(selected_item: list, item:pd.DataFrame, user:pd.DataFrame):
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
        subplot_titles=("Age(profile)", "Age(rec)"),
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
         template='ggplot2'
        # title_text="Selected Item profile vs Selected Item rec list (Age)",
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
        subplot_titles=("Gender(profile)", "Gender(rec)"),
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
         template='ggplot2'
        # title_text="Selected Item profile vs Selected Item rec list (Gender)",
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
        subplot_titles=("Occupation(profile)", "Occupation(rec)"),
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
         template='ggplot2'
        # title_text="Selected Item profile vs Selected Item rec list (Occupation)",
        # width=1000,
        # height=500
    )

    return fig

# 유저 세 번째 사이드 정보
def plot_usergroup_genre(item, origin_item, rerank_item, profile_item, tmp):
            """
            need variance :
            dataset = dataset.info class
            run1 = model_managers['EASE'].get_all_model_configs()[0] 과 같은 실험 정보
            user_df = pd.merge(dataset.user_df, dataset.ground_truth, on='user_id')
            중복 아이템을 셀지 말지의 여부를 가르는 문제가 있다.
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
                text=f"Total users num in this group : {len(tmp)}",
                x=0.5,
                y=0.5,
                font_size=20,
                showarrow=False,
            )
            fig.update_layout(
                # title_text=f"User group genre pie chart",
                width=1000,
                height=800,
                template='ggplot2'
            )

            return fig


def plot_info_counter(Counter_profile: Counter, info_name:str, k:int=10):
        """
        임베딩 그래프 옆에 사이드 정보 그려주는 함수

        Counter_profile:
        info_name: 그래프에 출력될 문자열
        k: 파이 차트로 보여줄 원소 갯수. 너무 많으면 보기 안 좋기에 적당히 설정
        """
        Counter_profile_labels = list(Counter_profile.keys())[:k]
        Counter_profile_values = list(Counter_profile.values())[:k]
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "domain"}]],
            subplot_titles=(f"{info_name} ratio(profile)"),
        )
        fig.add_trace(
            go.Pie(
                labels=Counter_profile_labels,
                values=Counter_profile_values,
                name=f"{info_name}(profile)",
                pull=[0.07] + [0] * (len(Counter_profile_values) - 1),
            ),  # textinfo='label+percent', pull=[0.2]+[0]*(len(user_rec_values)-1)
            1,
            1,
        )
        fig.update_layout(
            template='ggplot2'
            )
        fig.update_traces(hole=0.3, hoverinfo="label+percent+name")
        return fig