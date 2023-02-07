import dash
import json
import copy
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import dash_bootstrap_components as dbc

from dash import html, dcc, callback, Input, Output, State,  MATCH, ALL
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
from .utils import global_component as gct

dash.register_page(__name__, path='/reranking')

rerank_total_metrics = None
rerank_total_metrics_users = None
SPECIFIC_PLOT_WIDTH = 1000

rerank_metric_option = [
    {'label':' Diversity(cosine)', 'value':'diversity(cos)'},
    {'label':' Diversity(jaccard)', 'value':'diversity(jac)'},
    {'label':' Serendipity(PMI)', 'value':'serendipity(pmi)'},
    {'label':' Serendipity(jaccard)', 'value':'serendipity(jac)'},
    {'label':' Novelty', 'value':'novelty'},
]

rerank_metric_list = [
    'diversity(cos)',
    'diversity(jac)',
    'serendipity(pmi)',
    'serendipity(jac)',
    'novelty'
]



### alpha 선택하는 부분, radio
alpha_info = html.Div([
    dcc.Markdown(id="user_alpha", dangerously_allow_html=True),
    # dbc.RadioItems(
    #         id="alpha",
    #         className="btn-group",
    #         inputClassName="btn-check",
    #         labelClassName="btn btn-outline-primary",
    #         labelCheckedClassName="active",
    #         options=[
    #             {"label": "0.5", "value": 0.5}, # TODO: 사용자가 지정한 alpha값으로 
    #             # {"label": "1", "value": 1, "disabled":True},  # 사용자가 지정한 alpha로 지정
    #         ],
    #         value=0.5,
    #                     ),
], className="radio-group mb-2 mt-0")

model_form = html.Div([
    html.H6(["실험 선택"], id="test431", className="mb-2"),
    dcc.Dropdown(id="selected_model_by_name", placeholder=""),
            html.Hr(),
            html.H6(["Reranking 파라미터 α ", html.Span("�", id="alpha-tooltip")], className="mb-0"),
            dbc.Tooltip(["일종의 가중치이며, 낮을수록 Reranking 모델의 추천이 많이 반영됩니다.",
                         html.Span("FAQ를 참고하세요!", id="alpha_faq_link")],
                     target="alpha-tooltip",
                    #  className="w-auto"
                     ),
            html.Br(),
            dcc.Markdown(id="user_alpha", dangerously_allow_html=True),
            html.Hr(),
            html.H6("목적 함수 선택"),
    dcc.Checklist(
        rerank_metric_option,
        rerank_metric_list,
        labelStyle={"display":"block"},
        id="obj_funcs",)
], className='form-style')

sidebar = html.Div(
    [
        html.H3("옵션 선택", className="mt-3", style={"margin-bottom":25}),
        html.Hr(),
        html.Div(id='rerank_form', children=model_form),
        dbc.Button('Rerank!', id="rerank_btn", n_clicks=0, className="mt-3")
    ],
    className='sidebar'
)

total_graph = html.Div([
    html.Br(),
    html.H1(children='Reranking', style={'text-align': 'center','font-weight': 'bold'}),
    html.Hr(),
    
    html.Div(id='reranked_graph'),
    
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Br(),
            ]),
            html.Div(id='rerank_total_metric')
            ]),
        ])
     ])

def specific_metric():
    specific_metric = html.Div([
        html.H3('세부 지표 분석'),
        dbc.Row([
            dbc.Col([
                dbc.RadioItems(
                    id="rerank_sort_of_metric",
                    className="btn-group mb-2",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-primary",
                    labelCheckedClassName="active",
                    options=[
                        {"label": "정성 지표", "value": 'Qual'},
                        {"label": "정량 지표", "value": 'Quant'},
                    ],
                    value='Qual',
                    
                ),
                html.Br(),
                dcc.Dropdown(id='rerank_metric_list', className="specific-metric")
                ], width=3),
                html.Br(),
                html.Div([html.P(id="print_metric"),]),
            html.Div([
                dcc.Graph(id = 'rerank_bar_fig'), #figure=fig_qual
                dbc.Col(dbc.Spinner(html.Div(id = 'rerank_dist_fig'), color="primary")),  # dcc.Graph(id = 'dist_fig')
            ], className="vstack")
        ]),

        ],
        className="radio-group",
    )
    return specific_metric


layout = html.Div(
    
    [
    gct.get_navbar(has_sidebar=False),
    html.Div([
    sidebar,
    total_graph,
    html.Div(id = 'rerank_specific_metric_children')
    ], className="content"),
    # html.Div(id='trash2'),
    dcc.Store(id='store_selected_exp', storage_type='session'),
    dcc.Store(id='store_exp_names', storage_type="session"),
    dcc.Store(id='store_exp_ids', storage_type='session'),
    dcc.Store(id='store_selected_exp_names', data=[], storage_type='session')

],)



### 어떤 실험이 선택 가능한지 store에서 가져옴 (실험의 이름으로)
@callback(
    Output("selected_model_by_name", "options"),
    Input("rerank_btn", "n_clicks"),
    State("store_exp_names", "data"),
)
def print_selected_exp_name(n, exp_name):
    if n != 0:
        PreventUpdate
    exp_name = copy.deepcopy(exp_name)
    exp_name = list(set(exp_name))
    return exp_name

### exp_names가 들어오면 original + rerank 실험 정보들 선언하고 alpha값 return 하는 callback
@callback(
    Output('user_alpha', 'children'),
    # Input('rerank_btn', 'n_clicks'),
    Input('selected_model_by_name', 'value'),
    State('store_selected_exp','data')
)
def get_stored_selected_models(exp_names:str, store) -> pd.DataFrame:
    if exp_names is None:
        return "위에서 실험을 선택해주세요."

    ### original 실험 정보 가져오기
    store_df = pd.DataFrame(store).set_index('experiment_name')
    exp_id = store_df.loc[exp_names, 'exp_id']
    params = {'ID':'mkdir', 'dataset_name':'ml-1m', 'exp_ids': [exp_id]}
    response = requests.get(gct.API_URL + '/frontend/selected_metrics', params = params)
    a = response.json()
    exp_metrics = pd.DataFrame().from_dict(a['model_metrics'], orient='tight')
    exp_metrics.index = ['original']
    exp_metrics = exp_metrics.rename_axis('objective_fn')
    exp_metrics_users = pd.DataFrame().from_dict(a['user_metrics'], orient='tight')
    exp_metrics_users.index = ['original']
    exp_metrics_users = exp_metrics_users.rename_axis('objective_fn')

    ### rerank 실험 정보 가져오기
    global rerank_total_metrics
    global rerank_total_metrics_users
    params = {'ID':'mkdir', 'dataset_name':'ml-1m', 'exp_names': [exp_names]}
    response = requests.get(gct.API_URL + '/frontend/reranked_exp', params = params)
    a = response.json()

    rerank_total_metrics = pd.DataFrame().from_dict(a['model_info'], orient='tight')
    rerank_total_metrics = rerank_total_metrics.set_index('objective_fn')
    # print(rerank_total_metrics)
    # rerank_total_metrics = rerank_total_metrics.loc['']
    rerank_total_metrics = rerank_total_metrics.drop(['experiment_name'], axis=1) # , 'alpha'
    rerank_total_metrics = pd.concat([exp_metrics, rerank_total_metrics], axis=0)

    rerank_total_metrics_users = pd.DataFrame().from_dict(a['user_metrics'], orient='tight')
    rerank_total_metrics_users.index = rerank_total_metrics.index[1:]
    rerank_total_metrics_users = pd.concat([exp_metrics_users, rerank_total_metrics_users], axis=0)

    print(rerank_total_metrics.loc['novelty','alpha'])
    return f"현재 실험에서 선택한 α : {rerank_total_metrics.loc['novelty','alpha']}"

### 사용자가 선택한 alpha 값이 무엇인지 표시해주는 callback
# @callback(
#         Output('alpha_user', 'children'),
#         State(),
#         Input(),
# )
# def print_alpha():
#     rerank_total_metrics['alpha'] = 


### rerank! 버튼을 누르면 plot을 그려주는 callback
@callback(  # rerank 버튼 누름
        Output('rerank_specific_metric_children', 'children'),
        Output('rerank_total_metric', 'children'),
        Input('store_selected_exp_names', 'data'),
        Input('rerank_btn', 'n_clicks'),
        State('obj_funcs','value'),
        State('rerank_btn', 'n_clicks'),
        State('store_selected_exp','data')
        # prevent_initial_call=True
)
def plot_total_metrics(data, n, obj_funcs, state, store): # df:pd.DataFrame
    if state == 0:
        return html.Div([]), dbc.Alert("Rerank 버튼을 눌러 리랭킹 된 실험들의 지표를 확인해보세요!", color="info")
        # html.Div([
        #     html.P("If you want to metric compare between selected models, Click Compare!"),
        # ])
    else:
        # 모델간 정량, 정성 지표 plot (Compare Table에 있는 모든 정보들 활용)
        colors = ['#9771D0', '#D47DB2', '#5C1F47', '#304591', '#BAE8C8', '#ECEBC6', '#3D3D3D'] # 사용자 입력으로 받을 수 있어야 함
        store_df = pd.DataFrame(store).set_index('experiment_name')
        tmp_metrics = rerank_total_metrics.drop(['diversity_jac','serendipity_jac','alpha'], axis=1)
        obj_funcs = ['original'] + obj_funcs
        tmp_metrics = tmp_metrics.loc[obj_funcs]
        metrics = list(tmp_metrics.columns)
        fig = go.Figure()
        for i,obj_fn in enumerate(tmp_metrics.index):  # data
            # exp_id = store_df.loc[exp_name, 'exp_id'] # exp_name에 맞는 exp_id 찾아주기
            fig.add_bar(name=obj_fn, x=metrics, y=list(tmp_metrics.loc[obj_fn,:]), text=list(tmp_metrics.loc[obj_fn,:]),) #  marker={'color' : colors[i]}
            # .apply(eval)은 np.array나 list를 문자열로 인식할 때만 활용해주면 됨
            # 아니면 TypeError: eval() arg 1 must be a string, bytes or code object 발생
        fig.update_layout(
            barmode='group',
            bargap=0.15, # gap between bars of adjacent location coordinates.
            bargroupgap=0.1, # gap between bars of the same location coordinate.)
            template='ggplot2'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

        return specific_metric(), [html.H3('전체 지표 정보'),dcc.Graph(figure=fig)]  # id = 'total_metric'


### metric lists를 보여주는 callback
@callback(
    Output('rerank_metric_list', 'options'),
    Input('rerank_sort_of_metric', 'value'),
)
def load_metric_list(sort_of_metric:str) -> list:
    if sort_of_metric == 'Quant':
        metric_list = [
            {'label': 'Recall_k', 'value' : 'recall'},
            {'label':'AP@K', 'value':'map'},
            {'label':'NDCG', 'value':'ndcg'},
            {'label':'TailPercentage', 'value':'tail_percentage'},
            {'label':'AvgPopularity', 'value':'avg_popularity'},
            ]
    elif sort_of_metric == 'Qual':
        metric_list = [
            {'label':'Diversity(cosine)', 'value':'diversity_cos'},
            {'label':'Diversity(jaccard)', 'value':'diversity_jac'},
            {'label':'Serendipity(PMI)', 'value':'serendipity_pmi'},
            {'label':'Serendipity(jaccard)', 'value':'serendipity_jac'},
            {'label':'Novelty', 'value':'novelty'},
            ]
    return metric_list

@callback(
    Output('rerank_bar_fig', 'figure'),
    State('store_selected_exp_names', 'data'),
    State('obj_funcs','value'),
    Input("rerank_sort_of_metric", 'value'),
    State('store_selected_exp','data')
)
def plot_bar(data, obj_funcs, sort_of_metric, store):
    store_df = pd.DataFrame(store).set_index('experiment_name')
    colors = ['#9771D0', '#D47DB2', '#5C1F47', '#304591', '#BAE8C8', '#ECEBC6', '#3D3D3D']
    obj_funcs = ['original'] + obj_funcs
    if sort_of_metric == 'Qual':
        qual_metrics = rerank_total_metrics.iloc[:,7:]
        qual_metrics = qual_metrics.drop(['alpha'], axis=1)
        qual_metrics = qual_metrics.loc[obj_funcs]
        metrics = list(qual_metrics.columns)

        fig = go.Figure()
        for i,obj_fn in enumerate(qual_metrics.index):  # data
            # exp_id = store_df.loc[exp_name, 'exp_id'] # exp_name에 맞는 exp_id 찾아주기
            fig.add_bar(name=obj_fn, x=metrics, y=list(qual_metrics.loc[obj_fn,:]), text=list(qual_metrics.loc[obj_fn,:])) #, marker={'color' : colors[i]}

        fig.update_layout(
            barmode='group',
            bargap=0.15, # gap between bars of adjacent location coordinates.
            bargroupgap=0.1, # gap between bars of the same location coordinate.)
            title_text='전체 정성 지표',
            width = SPECIFIC_PLOT_WIDTH,
            template='ggplot2'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        return fig

    elif sort_of_metric == 'Quant':
        quant_metrics = rerank_total_metrics.iloc[:,1:7]
        quant_metrics = quant_metrics.loc[obj_funcs]
        metrics = list(quant_metrics.columns)

        fig = go.Figure()
        for i,obj_fn in enumerate(quant_metrics.index):  # data
            # exp_id = store_df.loc[exp_name, 'exp_id'] # exp_name에 맞는 exp_id 찾아주기
            fig.add_bar(name=obj_fn, x=metrics, y=list(quant_metrics.loc[obj_fn,:]), text=list(quant_metrics.loc[obj_fn,:])) #, marker={'color' : colors[i]}

        fig.update_layout(
            barmode='group',
            bargap=0.15, # gap between bars of adjacent location coordinates.
            bargroupgap=0.1, # gap between bars of the same location coordinate.)
            title_text='전체 정량 지표',
            width=SPECIFIC_PLOT_WIDTH,
            template='ggplot2'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
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

### 선택한 metric에 대한 dist plot을 띄워주는 callback
@callback(
    Output('rerank_dist_fig', 'children'),
    State('obj_funcs','value'),
    Input("rerank_metric_list", 'value'),
)
def plot_dist(obj_funcs, value):
    colors = ['#9771D0', '#D47DB2', '#5C1F47', '#304591', '#BAE8C8', '#ECEBC6', '#3D3D3D']
    obj_funcs = ['original'] + obj_funcs
    if value in ['diversity_jac', 'diversity_cos', 'serendipity_pmi', 'serendipity_jac', 'novelty']:
        group_labels = obj_funcs
        colors = colors[:len(obj_funcs)]
        hist_data = rerank_total_metrics_users.loc[obj_funcs, value].values
        fig = go.Figure()
        for each in hist_data:
            fig.add_trace(go.Histogram(x=each, nbinsx=100))
        # fig = ff.create_distplot(np.array(hist_data), group_labels, # colors=colors,
        #                         bin_size=0.025, show_rug=True, curve_type='kde')

        metric_list = {
            "diversity_cos": "Diversity(cosine)",
            "diversity_jac": "Diversity(jaccard)",
            'serendipity_pmi':'Serendipity(PMI)',
            'serendipity_jac':'Serendipity(jaccard)',
            'novelty':'Novelty',
            }
        fig.update_layout(title_text=f'유저별 {metric_list[value]} 분포',
                          barmode='overlay',
                          width=SPECIFIC_PLOT_WIDTH, template='ggplot2')
        fig.update_traces(opacity=0.6)
        return dcc.Graph(figure=fig)

    elif value in ['recall', 'ndcg', 'map', 'avg_popularity', 'tail_percentage']:
        if value == 'map':
            value = 'avg_precision'
        group_labels = obj_funcs
        colors = colors[:len(obj_funcs)]
        hist_data = rerank_total_metrics_users.loc[obj_funcs, value].values
        fig = go.Figure()
        for each in hist_data:
            fig.add_trace(go.Histogram(x=each, nbinsx=100))
        # fig = ff.create_distplot(hist_data, group_labels, # colors=colors,
        #                         bin_size=0.025, show_rug=True, curve_type='kde')

        metric_list = {
            'recall':'Recall@k',
            "avg_precision": "AP@K",
            "ndcg": "NDCG",
            "tail_percentage":"TailPercentage",
            "avg_popularity":"AvgPopularity",
            }                                
        fig.update_layout(title_text=f'유저별 {metric_list[value]} 분포',
                          barmode='overlay',
                          width=SPECIFIC_PLOT_WIDTH, template='ggplot2')
        fig.update_traces(opacity=0.6)
        return dcc.Graph(figure=fig)
    else:
        return html.Div([])