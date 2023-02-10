import dash
import json
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import dash_bootstrap_components as dbc

from dash import html, dcc, callback, Input, Output, State, MATCH, ALL
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
from .utils import global_component as gct

dash.register_page(__name__, path="/model-vs-model")

total_metrics = None
total_metrics_users = None
SPECIFIC_PLOT_WIDTH = 1200
sidebar = html.Div(
    [
        html.H3("실험 선택", className="mt-3", style={"margin-bottom": 21}),
        html.Hr(),
        html.Div(id="model_form", children=[]),
        html.Div(
            [
                dbc.Button(
                    "➕",
                    id="add_button",
                    n_clicks=0,
                    # style={'text-color':'#2E8B57'},
                    className="mt-1 me-5",
                ),
                dbc.Tooltip(
                    "실험 추가하기",
                    target="add_button",
                    placement="bottom",
                    style={"width": 120},
                ),
                # dbc.Popover("실험 추가하기", trigger='hover', target='add_button', body=True, placement='bottom'),
                dbc.Button(
                    "비교하기", id="compare_btn", n_clicks=0, className="ms-5 mt-1 w-50"
                ),
            ],
            className="hstack gap-3",
        ),
    ],
    className="sidebar",
)

#### total metric 그래프 그릴 부분
total_graph = html.Div(
    [
        html.Div(id="select_model2"),
        html.Div(id="total_metric"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div([]),
                        # html.Div(id='total_metric')
                        # dcc.Graph(id='total_metric') # html.Div(id='total_metric')
                    ]
                ),
            ]
        ),
    ]
)

#### 정량, 정성 지표 그래프 그릴 부분
def specific_metric():
    specific_metric = html.Div(
        [
            html.H3("세부 지표 분석"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.RadioItems(
                                id="sort_of_metric",
                                className="btn-group mb-2",
                                inputClassName="btn-check",
                                labelClassName="btn btn-outline-primary",
                                labelCheckedClassName="active",
                                options=[
                                    {"label": "정성 지표", "value": "Qual"},
                                    {"label": "정량 지표", "value": "Quant"},
                                ],
                                value="Qual",
                            ),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="metric_list",
                                            className="specific-metric",
                                            placeholder="지표를 선택하세요",
                                        )
                                    ),
                                    dbc.Col(html.Div(id="metric_info_message")),
                                ],
                                className="hstack gap-3",
                            ),
                        ],
                        width=60,
                    ),
                    html.Br(),
                    html.Div(
                        [
                            html.P(id="print_metric"),
                        ]
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="bar_fig"),
                            dbc.Col(
                                dbc.Spinner(html.Div(id="dist_fig"), color="primary")
                            ),
                        ],
                        className="vstack",
                    )
                    # dbc.Col([
                    #     dbc.Row([
                    #         dbc.Col(dcc.Graph(id = 'bar_fig')), #figure=fig_qual),
                    #         dbc.Col(html.Div(id = 'dist_fig')), #figure=fig_qual),
                    #      # dcc.Graph(id = 'dist_fig')
                    #     ])
                    # ], width=8,)
                ]
            ),
        ],
        className="radio-group",
    )
    return specific_metric


layout = html.Div(
    [
        html.Div(
            [
                gct.get_navbar(has_sidebar=False),
            ]
        ),
        html.Div(
            [
                sidebar,
                html.Br(),
                # html.H1(children='Model vs Model', style={'text-align': 'center','font-weight': 'bold'}),
                # html.Hr(),
                total_graph,
                html.Div(id="specific_metric_children"),
            ],
            className="content w-75",
        ),
        html.Div(id="trash"),
        dcc.Store(id="store_selected_exp", storage_type="session"),
        dcc.Store(id="store_exp_names", storage_type="session"),
        dcc.Store(id="store_exp_ids", storage_type="session"),
        dcc.Store(id="store_selected_exp_names", data=[], storage_type="session"),
    ]
)


### exp_ids가 들어오면 실험 정보들 return 하는 callback
@callback(
    Output("trash", "children"),
    Input("compare_btn", "n_clicks"),
    State("store_exp_ids", "data"),
    State("store_user_state", "data"),
    State("store_user_dataset", "data"),
)
def get_stored_selected_models(_, exp_ids: list[int], user, dataset) -> pd.DataFrame:
    global total_metrics
    global total_metrics_users
    # params = {'ID':'mkdir', 'dataset_name':'ml-1m', 'exp_ids': exp_ids}
    params = {"ID": user["username"], "dataset_name": dataset, "exp_ids": exp_ids}
    response = requests.get(gct.API_URL + "/frontend/selected_metrics", params=params)
    a = response.json()
    total_metrics = pd.DataFrame().from_dict(a["model_metrics"], orient="tight")
    total_metrics_users = pd.DataFrame().from_dict(a["user_metrics"], orient="tight")
    return html.Div([])


# @callback(
#         Output(),
#         Input("store_exp_names", "data")
# )

### 어떤 실험을 고를지 select하는 dropdown을 보여주는 callback
@callback(
    Output("model_form", "children"),
    [
        Input("add_button", "n_clicks"),
        Input({"type": "delete_btn", "index": ALL}, "n_clicks"),
        Input("store_exp_names", "data"),
    ],
    [State("model_form", "children")],
)
def display_dropdowns(n_clicks, _, store_exp_names, children):
    if store_exp_names == None:
        raise PreventUpdate
    input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if "index" in input_id:
        delete_chart = json.loads(input_id)["index"]
        children = [
            chart
            for chart in children
            if "'index': " + str(delete_chart) not in str(chart)
        ]
    else:
        model_form = html.Div(
            [
                dbc.Button(
                    "➖", className="mb-3", id={"type": "delete_btn", "index": n_clicks}
                ),
                # dbc.Popover("Delete this experiment", trigger='hover', target={'type':'delete_btn', 'index':ALL}, body=True), # 동적 컴포넌트에는 어떻게 적용해야 할지 모르겠음
                dcc.Dropdown(
                    store_exp_names,
                    id={"type": "selected_exp", "index": n_clicks},
                    placeholder="실험을 선택하세요.",
                    optionHeight=50,  # options=[{'label':'exp_name', 'value':'exp_id'}, ... ], # label은 보여지는거, value는 실제 어떤 데이터인지
                ),
                html.Hr(),
                dcc.Markdown(
                    id={"type": "exp's_hype", "index": n_clicks},
                    dangerously_allow_html=True,
                ),
                html.Br(),
            ],
            className="form-style my-2",
        )
        children.append(model_form)
    return children


### selected_exp 의 hype을 소개하는 callback
@callback(
    Output({"type": "exp's_hype", "index": MATCH}, "children"),
    [
        Input({"type": "selected_exp", "index": MATCH}, "value"),
        State("store_selected_exp", "data"),
    ],
)
def display_output(selected_dropdown: str, data) -> str:  #
    if selected_dropdown is None:
        raise PreventUpdate

    tmp_df = pd.DataFrame(data).set_index("experiment_name")
    exp_hype = tmp_df.loc[selected_dropdown, "hyperparameters"]
    exp_hype = eval(exp_hype)
    seq = ""
    #한 파라미터씩 완성하기. []는 들어가면 안 되기에 수정
    #라이브러리 단에서 이후 발생할 예외는 라이브러리 단에서 수정하는 게 좋을 수도
    for i in exp_hype:
        seq += str(i) + " : " + str(exp_hype[i]).replace("[","").replace("]","") + "<br>"
    return seq


### selected_exp의 experiment_name을 저장
@callback(
    Output("store_selected_exp_names", "data"),
    Input({"type": "selected_exp", "index": ALL}, "value"),
    State("store_selected_exp_names", "data"),
)
def save_selected_exp_names(value, data):
    if value == [None]:
        raise PreventUpdate
    return value


### compare! 버튼을 누르면 plot을 그려주는 callback
@callback(  # compare 버튼 누름
    Output("specific_metric_children", "children"),
    Output("total_metric", "children"),
    Input("store_selected_exp_names", "data"),
    Input("compare_btn", "n_clicks"),
    State("compare_btn", "n_clicks"),
    State("store_selected_exp", "data")
    # prevent_initial_call=True
)
def plot_total_metrics(data, n, state, store):  # df:pd.DataFrame
    if state == 0:
        return html.Div([]), dbc.Alert(
            [
                "왼쪽에서 모델을 선택하고 '비교하기' 버튼을 눌러 실험들의 지표를 그래프로 확인해보세요! ",
                # html.Span("(2개 이상부터 가능합니다.)", className="fw-bold")
            ],
            color="info",
            style={"width": "80%"},
        )
    else:
        # 모델간 정량, 정성 지표 plot (Compare Table에 있는 모든 정보들 활용)
        colors = [
            "#9771D0",
            "#D47DB2",
            "#5C1F47",
            "#304591",
            "#BAE8C8",
            "#ECEBC6",
            "#3D3D3D",
        ]  # 사용자 입력으로 받을 수 있어야 함
        store_df = pd.DataFrame(store).set_index("experiment_name")
        tmp_metrics = total_metrics.drop(["diversity_jac", "serendipity_pmi"], axis=1)
        metrics = list(tmp_metrics.columns)
        fig = go.Figure()
        for i, exp_name in enumerate(data):
            exp_id = store_df.loc[exp_name, "exp_id"]  # exp_name에 맞는 exp_id 찾아주기
            fig.add_bar(
                name=exp_name,
                x=metrics,
                y=list(tmp_metrics.loc[exp_id, :]),
                text=list(tmp_metrics.loc[exp_id, :]),
            )  # , marker={'color' : colors[i]}
            # .apply(eval)은 np.array나 list를 문자열로 인식할 때만 활용해주면 됨
            # 아니면 TypeError: eval() arg 1 must be a string, bytes or code object 발생
        fig.update_layout(
            yaxis_range=[0, 1.05],
            barmode="group",
            bargap=0.25,  # gap between bars of adjacent location coordinates.
            bargroupgap=0.1,  # gap between bars of the same location coordinate.)
            template="ggplot2"
            # title_text='Metric indicators'
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        child = html.Div(
            dbc.Row(
                [
                    dbc.Col(html.H3("전체 지표 정보")),
                    dbc.Col(
                        dbc.Alert(
                            "아래 그래프를 확인하며 사용할 모델을 결정하고, Reranking / Deep Analysis 페이지로 이동하세요!",
                            # html.Span("(2개 이상부터 가능합니다.)", className="fw-bold")
                            color="warning",
                            style={
                                "width": "700px",
                                "margin-right": "0",
                                "margin-left": "260px",
                            },
                        )
                    ),
                ]
            )
        )
        return specific_metric(), [child, dcc.Graph(figure=fig)]


### metric lists를 보여주는 callback
@callback(
    Output("metric_list", "options"),
    Input("sort_of_metric", "value"),
)
def load_metric_list(sort_of_metric: str) -> list:
    if sort_of_metric == "Quant":
        metric_list = [
            {"label": "Recall@k", "value": "recall"},
            {"label": "AP@K", "value": "map"},
            {"label": "NDCG", "value": "ndcg"},
            {"label": "TailPercentage", "value": "tail_percentage"},
            {"label": "AvgPopularity", "value": "avg_popularity"},
        ]
    elif sort_of_metric == "Qual":
        metric_list = [
            {"label": "Diversity(cosine)", "value": "diversity_cos"},
            {"label": "Diversity(jaccard)", "value": "diversity_jac"},
            {"label": "Serendipity(PMI)", "value": "serendipity_pmi"},
            {"label": "Serendipity(jaccard)", "value": "serendipity_jac"},
            {"label": "Novelty", "value": "novelty"},
        ]
    return metric_list


### Qual or Quant 선택하면 metric bar plot 띄워주는 Callback
@callback(
    Output("bar_fig", "figure"),
    State("store_selected_exp_names", "data"),
    Input("sort_of_metric", "value"),
    State("store_selected_exp", "data"),
)
def plot_bar(data, sort_of_metric, store):
    store_df = pd.DataFrame(store).set_index("experiment_name")
    colors = [
        "#9771D0",
        "#D47DB2",
        "#5C1F47",
        "#304591",
        "#BAE8C8",
        "#ECEBC6",
        "#3D3D3D",
    ]
    if sort_of_metric == "Qual":
        qual_metrics = total_metrics.iloc[:, 6:]
        metrics = list(qual_metrics.columns)

        fig = go.Figure()
        for i, exp_name in enumerate(data):
            exp_id = store_df.loc[exp_name, "exp_id"]  # exp_name에 맞는 exp_id 찾아주기
            fig.add_bar(
                name=exp_name,
                x=metrics,
                y=list(qual_metrics.loc[exp_id, :]),
                text=list(qual_metrics.loc[exp_id, :]),
            )  # , marker={'color' : colors[i]}

        fig.update_layout(
            yaxis_range=[0, 1.05],
            barmode="group",
            bargap=0.25,  # gap between bars of adjacent location coordinates.
            bargroupgap=0.1,  # gap between bars of the same location coordinate.)
            title_text="전체 정성 지표",
            width=SPECIFIC_PLOT_WIDTH,
            template="ggplot2",
            font=dict(
                size=18,
            ),
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        return fig

    elif sort_of_metric == "Quant":
        quant_metrics = total_metrics.iloc[:, :6]
        metrics = list(quant_metrics.columns)

        fig = go.Figure()
        for i, exp_name in enumerate(data):
            exp_id = store_df.loc[exp_name, "exp_id"]  # exp_name에 맞는 exp_id 찾아주기
            fig.add_bar(
                name=exp_name,
                x=metrics,
                y=list(quant_metrics.loc[exp_id, :]),
                text=list(quant_metrics.loc[exp_id, :]),
            )  # , marker={'color' : colors[i]}

        fig.update_layout(
            yaxis_range=[0, 1.05],
            barmode="group",
            bargap=0.25,  # gap between bars of adjacent location coordinates.
            bargroupgap=0.1,  # gap between bars of the same location coordinate.)
            title_text="전체 정량 지표",
            width=SPECIFIC_PLOT_WIDTH,
            template="ggplot2",
            font=dict(
                size=18,
            ),
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        return fig

    else:
        return html.Div([])


# ### 선택한 metric 뭔지 보여주는 test callback
# @callback(
#     Output('print_metric', 'children'),
#     Input("metric_list", 'value'),
# )
# def print_metric(value):
#     return f'user selection : {value}'

### 선택한 metric에 대한 dist plot을 띄워주는 callback, 나중에 refactoring 가능
@callback(
    Output("dist_fig", "children"),
    Output("metric_info_message", "children"),
    State("store_selected_exp_names", "data"),
    Input("metric_list", "value"),
    State("store_selected_exp", "data"),
)
def plot_dist(data, value, store):
    colors = [
        "#9771D0",
        "#D47DB2",
        "#5C1F47",
        "#304591",
        "#BAE8C8",
        "#ECEBC6",
        "#3D3D3D",
    ]
    store_df = pd.DataFrame(store).set_index("experiment_name")
    selected_id = store_df.loc[data, "exp_id"].values  # 선택한 실험만 보여줘야 함
    metric_info_list = dbc.Alert(
        "아래로 스크롤해 선택 지표의 distribution을 확인해보세요",
        color="info",
        className="w-500",
        style={"width": "100%"},
    )
    if value in [
        "diversity_jac",
        "diversity_cos",
        "serendipity_pmi",
        "serendipity_jac",
        "novelty",
    ]:
        group_labels = data
        colors = colors[: len(data)]
        hist_data = total_metrics_users.loc[selected_id, value].values
        fig = go.Figure()
        for each in hist_data:
            fig.add_trace(go.Histogram(x=each, nbinsx=100))
        # fig = ff.create_distplot(hist_data, group_labels, # colors=colors,
        #                         bin_size=0.025, show_rug=True, curve_type='kde')

        metric_list = {
            "diversity_jac": "Diversity(jaccard)",
            "diversity_cos": "Diversity(cosine)",
            "serendipity_jac": "Serendipity(jaccard)",
            "serendipity_pmi": "Serendipity(PMI)",
            "novelty": "Novelty",
        }
        fig.update_layout(
            title_text=f"유저별 {metric_list[value]} 분포",
            barmode="overlay",
            width=SPECIFIC_PLOT_WIDTH,
            template="ggplot2",
            font=dict(
                size=18,
            ),
        )
        fig.update_traces(opacity=0.6)
        return dcc.Graph(figure=fig), metric_info_list

    elif value in ["recall", "ndcg", "map", "avg_popularity", "tail_percentage"]:
        if value == "map":
            value = "avg_precision"
        group_labels = data
        colors = colors[: len(data)]
        hist_data = total_metrics_users.loc[selected_id, value].values

        fig = go.Figure()
        for each in hist_data:
            fig.add_trace(go.Histogram(x=each, nbinsx=100))
        # fig = ff.create_distplot(hist_data, group_labels, # colors=colors,
        #                         bin_size=0.025, show_rug=True, curve_type='kde')

        metric_list = {
            "recall": "Recall@k",
            "ndcg": "NDCG",
            "avg_precision": "AP@K",
            "avg_popularity": "AvgPopularity",
            "tail_percentage": "TailPercentage",
        }

        fig.update_layout(
            title_text=f"유저별 {metric_list[value]} 분포",
            barmode="overlay",
            width=SPECIFIC_PLOT_WIDTH,
            template="ggplot2",
            font=dict(
                size=18,
            ),
        )
        fig.update_traces(opacity=0.6)
        return dcc.Graph(figure=fig), metric_info_list
    else:
        return html.Div([]), html.Div([])
