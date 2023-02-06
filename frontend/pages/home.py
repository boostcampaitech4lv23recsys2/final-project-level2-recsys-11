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
                        html.H1(emoji, className="text-start"),
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

img_url = "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a9401680-b68f-4c66-898b-bf51d3e93c6b/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230206%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230206T074253Z&X-Amz-Expires=86400&X-Amz-Signature=7f5bb4bffb9f39294c4e24f601d52a569c98ba0e14aa4cf3066c4e1a8323a872&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject"

feature_compare_table = get_feature_component(
    img_url=img_url,
    emoji="📋",
    title="각 실험 별 지표 비교 테이블",
    description="사용자가 실험한 모델의 정성, 정량 지표를 계산한 결과를 테이블로 확인할 수 있습니다.",
    img_position="left"
)

img_url = "https://user-images.githubusercontent.com/76675506/216320888-7b790e97-61af-442c-93b3-c574ed0c119e.png"
feature1 = get_feature_component(
    img_url=img_url,
    emoji="🔍",
    title="모델의 임베딩을 다양한 관점에서 살펴볼 수 있습니다.",
    description="사용자가 선택한 옵션에 따라, 임베딩 그래프를 인터랙티브하게 변화시킬 수 있습니다.",
    img_position="right"
)

img_url = "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/61a511fc-932a-4cf9-b369-2fe9e1d940ab/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230206%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230206T074706Z&X-Amz-Expires=86400&X-Amz-Signature=a105df63067c112eda7ef4a86430965043611e59a9a87d5b6710d01f1bf6ca3e&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject"
feature2 = get_feature_component(
    img_url=img_url,
    emoji="😊",
    title="실험별 지표 비교와 리랭킹 기능을 사용할 수 있습니다.",
    description="다양한 실험의 지표를 한 눈에 그래프로 비교할 수 있으며, 리랭킹을 통해 다양한 정성지표 값을 상승 시킬 수 있습니다.",
    img_position="left"
)

problem_intro = html.Div([
            html.H1("💡"),
            html.H3("추천시스템 문제는 조금 다릅니다."),
            html.H5("일반적으로 AI 모델에서는 높은 정확성, 혹은 재현율이 서비스 사용자의 만족으로 이어집니다."),
            html.H5("하지만 추천시스템에서는 그렇지 않습니다.", className="fst-italic"),
            html.H5(["정량적 지표인 정확성, 재현율 외에도 ", html.Span("정성적 지표", className="text-danger"), "(다양성, 참신성 의외성 등)를 같이 고려해야 합니다."], ),
])
layout = html.Div([
    dbc.NavbarSimple([
        dbc.NavItem(dcc.Link(dbc.Button("시작하기!", className=" fs-6 mt-1", color="light"),href="/login"),),
        ], brand=gct.BRAND_LOGO, brand_style={"margin-left":'45%', 'font-size':"2rem", 'color':'#FFFAF0'}
        , color="primary", class_name="home-navbar", sticky="top", fluid=True),
    html.Div([
        html.Div([
            html.Div([
            html.H1('프로젝트 소개', className="pt-3 text-center fs-1 mx-auto"),
            # dcc.Link(dbc.Button("시작하기!", className=" fs-6 mt-3 mb-4", color="light"),href="/login"),
            ], className="hstack"),
            html.Hr(),
            
            problem_intro,
            
            html.H4([gct.BRAND_LOGO+'은 이를 해결할 수 있는 ', html.Span('실험 관리 툴', className="text-info"),'입니다.'], className="mt-5 pt-4 pb-3 text-center fs-1"),
            html.Hr(),
            feature_compare_table,
            feature1,
            feature2,
            html.Br(),
            dcc.Link(dbc.Button("시작하기!", className="position-absolute top-105 start-50 translate-middle w-25 fs-2 my-4"), href="/login"),
            html.Br(className="h-25")
            ], className="container position-relative pb-4"),
    ], className="pb-5", style={'background-color': '#FFFAF0'}),
])