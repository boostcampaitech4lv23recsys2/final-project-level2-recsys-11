import dash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from .utils import global_component as gct

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

img_url = "https://user-images.githubusercontent.com/67850665/217193988-a3b45bcd-ce8c-4e89-a8be-f95ae8f6f433.png"

feature_compare_table = get_feature_component(
    img_url=img_url,
    emoji="π",
    title="κ° μ€ν λ³ μ§ν λΉκ΅ νμ΄λΈ",
    description="μ¬μ©μκ° μ€νν λͺ¨λΈμ μ μ±, μ λ μ§νλ₯Ό κ³μ°ν κ²°κ³Όλ₯Ό νμ΄λΈλ‘ νμΈν  μ μμ΅λλ€.",
    img_position="left"
)


workflow_img_url = "https://user-images.githubusercontent.com/55279227/217210198-8df93292-b18d-4ee0-81dc-03f768e88c39.jpg"

img_url = "https://user-images.githubusercontent.com/76675506/216320888-7b790e97-61af-442c-93b3-c574ed0c119e.png"
feature1 = get_feature_component(
    img_url=img_url,
    emoji="π",
    title="λͺ¨λΈμ μλ² λ©μ λ€μν κ΄μ μμ μ΄ν΄λ³Ό μ μμ΅λλ€.",
    description="μ¬μ©μκ° μ νν μ΅μμ λ°λΌ, μλ² λ© κ·Έλνλ₯Ό μΈν°λν°λΈνκ² λ³νμν¬ μ μμ΅λλ€.",
    img_position="right"
)

img_url = "https://user-images.githubusercontent.com/67850665/217196527-9699cd4d-df74-42fd-8a7d-61030cbf0ef4.png"
feature2 = get_feature_component(
    img_url=img_url,
    emoji="π",
    title="μ€νλ³ μ§ν λΉκ΅μ λ¦¬λ­νΉ κΈ°λ₯μ μ¬μ©ν  μ μμ΅λλ€.",
    description="λ€μν μ€νμ μ§νλ₯Ό ν λμ κ·Έλνλ‘ λΉκ΅ν  μ μμΌλ©°, λ¦¬λ­νΉμ ν΅ν΄ λ€μν μ μ±μ§ν κ°μ μμΉ μν¬ μ μμ΅λλ€.",
    img_position="left"
)

problem_intro = html.Div([
            html.H1("π‘"),
            html.H3("μΆμ²μμ€ν λ¬Έμ λ μ‘°κΈ λ€λ¦λλ€."),
            html.H5("μΌλ°μ μΌλ‘ AI λͺ¨λΈμμλ λμ μ νμ±, νΉμ μ¬νμ¨μ΄ μλΉμ€ μ¬μ©μμ λ§μ‘±μΌλ‘ μ΄μ΄μ§λλ€."),
            html.H5("νμ§λ§ μΆμ²μμ€νμμλ κ·Έλ μ§ μμ΅λλ€.", className="fst-italic"),
            html.H5(["μ λμ  μ§νμΈ μ νμ±, μ¬νμ¨ μΈμλ ", html.Span("μ μ±μ  μ§ν", className="text-danger"), "(λ€μμ±, μ°Έμ μ±, μμΈμ± λ±)λ₯Ό κ°μ΄ κ³ λ €ν΄μΌ ν©λλ€."], ),
])
layout = html.Div([
    dbc.NavbarSimple([
        dbc.NavItem(dcc.Link(dbc.Button("μμνκΈ°!", className=" fs-6 mt-1",style={'margin-right':'20%','width':'25rem'}, color="light"),href="/login"),),
        ], brand=gct.BRAND_LOGO, brand_style={"margin-left":'47%', 'font-size':"2rem", 'color':'#FFFAF0'}
        , color="primary", class_name="home-navbar", sticky="top", fluid=True),
    html.Div([
        html.Div([
            html.Div([
            html.H1('νλ‘μ νΈ μκ°', className="pt-3 text-center fs-1 mx-auto"),
            # dcc.Link(dbc.Button("μμνκΈ°!", className=" fs-6 mt-3 mb-4", color="light"),href="/login"),
            ], className="hstack"),
            html.Hr(),
            
            problem_intro,
            
            html.H4([gct.BRAND_LOGO+'μ μ΄λ₯Ό ν΄κ²°ν  μ μλ ', html.Span('μ€ν κ΄λ¦¬ ν΄', className="text-info"),'μλλ€.'], 
            className="mt-5 pt-4 pb-3 text-center fs-1"),
            dbc.Row([
                    dbc.Col([
                        html.Img(src=workflow_img_url, className="feature-image rounded-2 border border-primary border-4"),
                    ], width={'size':10, 'offset':1})
                    ], 
                className="center"),
            html.Hr(),
            feature_compare_table,
            feature1,
            feature2,
            html.Br(),
            dcc.Link(dbc.Button("μμνκΈ°!", className="position-absolute top-105 start-50 translate-middle w-25 fs-2 my-4"), href="/login"),
            html.Br(className="h-25")
            ], className="container position-relative pb-4"),
    ], className="pb-5", style={'background-color': '#FFFAF0'}),
])