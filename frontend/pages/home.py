import dash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from . import global_component as gct

dash.register_page(__name__, path='/')

def get_feature_component(img_url:str, emoji:str,
            title:str, description:str, img_position:str="left") -> dbc.Row:
    IMAGE_COL_WIDTH:int=7
    TEXT_COL_WIDTH:int=5
    feature_cn = "feature-image rounded-2 border border-primary border-4"
    if img_position == "left":
        return dbc.Row([
                    dbc.Col(
                        html.Img(src=img_url, className=feature_cn),
                    width=IMAGE_COL_WIDTH),
                    dbc.Col([
                        html.H1(emoji, className="text-end"),
                        html.H3(title),
                        html.H5(description),
                        ], width=TEXT_COL_WIDTH)
                    ], className="pt-4 pb-4")

    elif img_position == "right":
        return dbc.Row([
                    dbc.Col([
                        html.H1(emoji, className="text-start"),
                        html.H3(title),
                        html.H5(description)
                    ], width=TEXT_COL_WIDTH),
                    dbc.Col([
                        html.Img(src=img_url, className=feature_cn),
                    ], width=IMAGE_COL_WIDTH)
                        ], className="pt-4 pb-4")
    else:
        ValueError("img_url must be left or right")

img_url = "https://user-images.githubusercontent.com/76675506/216320888-7b790e97-61af-442c-93b3-c574ed0c119e.png"
feature1 = get_feature_component(
    img_url=img_url,
    emoji="🤔",
    title="모델의 임베딩을 다양한 관점에서 살펴볼 수 있습니다.",
    description="사용자가 선택한 옵션에 따라, 임베딩 그래프를 인터랙티브하게 변화시킬 수 있습니다.",
    img_position="left"
)
feature2 = get_feature_component(
    img_url=img_url,
    emoji="😊",
    title="기능 소제목",
    description="기능 설명",
    img_position="right"
)

layout = html.Div([
    dbc.NavbarSimple([
        # dbc.NavItem(dcc.Link(dbc.Button("평가 시작하기!", className="position-fixed top-50 end-0 translate-middle-y w-25 fs-2 mt-3 mb-4",),href="/login"),)
        dbc.NavItem(dcc.Link(dbc.Button("평가 시작하기!", className=" fs-6 mt-3 mb-4", color="light"),href="/login"),)
], color="primary", className="navbar ", sticky="top", brand="𝙒𝙚𝙗𝟰𝙍𝙚𝙘",),
    html.Div([
        html.Div([
            html.H1('추천을 평가할 땐, 𝙒𝙚𝙗𝟰𝙍𝙚𝙘', className="pt-4 pb-4 text-center fs-1"),
            feature1,
            feature2,
            dcc.Link(dbc.Button("평가 시작하기!", className="position-absolute top-100 start-50 translate-middle w-25 fs-2 mt-3 mb-4 "), href="/login"),
            html.Br(),
            html.Br(),
            html.Br(className="h-25")
            ], className="container position-relative pb-4"),
    ]),
])


