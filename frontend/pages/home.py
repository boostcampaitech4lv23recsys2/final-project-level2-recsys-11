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

img_url = "https://user-images.githubusercontent.com/76675506/216320888-7b790e97-61af-442c-93b3-c574ed0c119e.png"
feature_compare_table = get_feature_component(
    img_url=img_url,
    emoji="📋",
    title="각 실험 별 지표 비교 테이블",
    description="사용자가 실험한 모델의 정성, 정량 지표를 계산한 결과를 테이블로 확인할 수 있습니다.",
    img_position="left"
)

feature1 = get_feature_component(
    img_url=img_url,
    emoji="🔍",
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

problem_intro = html.Div([
            html.H1("🤔"),
            html.H3("추천시스템 문제는 조금 다릅니다."),
            html.H5("일반적으로 AI 모델에서는 높은 정확성, 혹은 재현율이 서비스 사용자의 만족으로 이어집니다."),
            html.H5("하지만 추천시스템에서는 그렇지 않습니다.", className="fst-italic"),
            html.H5(["정량적 지표인 정확성, 재현율 외에도 ", html.Span("정성적 지표", className="text-danger"), "(다양성, 참신성 의외성 등)를 같이 고려해야 합니다."], ),
])
layout = html.Div([
    dbc.NavbarSimple([
        dbc.NavItem(dcc.Link(dbc.Button("시작하기!", className=" fs-6 mt-3 mb-4 m-0", color="light"),href="/login"),),
<<<<<<< HEAD
        dbc.NavbarBrand("𝙒𝙚𝙗𝟰𝙍𝙚𝙘", class_name="fs-2text-center position-absolute top-50 start-50 translate-middle large-font")
        ], color="primary", className="navbar position-relative", sticky="top", brand="Web4Rec"),
=======
        dbc.NavbarBrand("𝙒𝙚𝙗𝟰𝙍𝙚𝙘", class_name="fs-2text-center position-absolute top-50 start-50 translate-middle")
        ], color="primary", className=" sticky  ms-n5", sticky="top", brand="𝙒𝙚𝙗𝟰𝙍𝙚𝙘", dark=True),
>>>>>>> MKDIR-157-model-vs-model-Design
    html.Div([
        html.Div([
            html.Div([
            html.H1('💡 프로젝트 소개', className="pt-4 text-center fs-1 mx-auto"),
            # dcc.Link(dbc.Button("시작하기!", className=" fs-6 mt-3 mb-4", color="light"),href="/login"),
            ], className="hstack"),
            html.Hr(),
            
            problem_intro,
            
            html.H4(['🔧 𝙒𝙚𝙗𝟰𝙍𝙚𝙘은 이를 해결할 수 있는 ', html.Span('실험 관리 툴', className="text-info"),'입니다.'], className="mt-5 pt-4 pb-4 text-center fs-1"),
            html.Hr(),
            feature_compare_table,
            feature1,
            feature2,
            dcc.Link(dbc.Button("시작하기!", className="position-absolute top-100 start-50 translate-middle w-25 fs-2 my-3 "), href="/login"),
            html.Br(className="h-25")
            ], className="container position-relative pb-4"),
    ], className="my-5"),
])