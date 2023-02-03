import dash
import json
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
from . import global_component as gct
API_url = 'http://127.0.0.1:30004'

dash.register_page(__name__, path='/model-vs-model')



# total_dict = requests.get(API_url + '/')
# total_df = pd.DataFrame(total_dict).from_dict(orient='tight')

# def make_total_metric_df(tmp_df:pd.DataFrame):
#     # tmp_dict = requests.get(API_url + '/')
#     total_metric_df = tmp_df.loc[] 
#     return total_metric_df

# def make_qual_metric_df(tmp_df:pd.DataFrame):
#     qual_metric_df = tmp_df.loc[]
#     return qual_metric_df

# def make_quan_metric_df(tmp_df:pd.DataFrame):
#     quan_metric_df = tmp_df.loc[]
#     return quan_metric_df


def plot_qual_metrics(df:pd.DataFrame):
    # 모델간 정량, 정성 지표 plot (Compare Table에 있는 모든 정보들 활용)
    metrics = list(df.columns[1:])
    colors = ['#A56CC1', '#A6ACEC', '#63F5EF', '#425FEF'] # 사용자 입력으로 받을 수 있어야 함
    
    fig = go.Figure()

    for i in df.index:
        fig.add_bar(name=df['Name'][i], x=metrics, y=list(df.iloc[i,1:].apply(eval).apply(np.mean)), text=list(df.iloc[i,1:].apply(eval).apply(np.mean)), marker={'color' : colors[i]})
        # .apply(eval)은 np.array나 list를 문자열로 인식할 때만 활용해주면 됨
        # 아니면 TypeError: eval() arg 1 must be a string, bytes or code object 발생
    fig.update_layout(
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1, # gap between bars of the same location coordinate.)
        title_text='Metric indicators'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

    return fig


def plot_dist_for_metrics(qual_df:pd.DataFrame, metric:str):
    assert metric in qual_df.columns, 'Metric is not in the column'
    hist_data = [eval(each)[0] for each in qual_df[metric]]
    group_labels = qual_df['Name'].values
    colors = ['#A56CC1', '#A6ACEC', '#63F5EF', '#425FEF'] # 사용자 입력으로 받을 수 있어야 함
    # colors = qual_df['colors]
    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                            bin_size=0.025, show_rug=True, curve_type='kde')

    # Add title
    fig.update_layout(title_text='Distribution of metrics')

    return fig

# 옵션으로 선택된 실험들을 불러옴
# 불러온 실험들로 df를 제작함
# 만약 새로운 실험이 + 되면 그 실험 정보를 df에 추가함 e.g.,) df.loc[] = ...

total_metrics = pd.read_csv('/opt/ml/total_metrics_df.csv')
# total_metrics = make_total_metrics_df()
qual_metrics = pd.read_csv('/opt/ml/qual_metrics_df.csv')  # 정성 지표들이 array로 담겨있음
# qual_metrics = make_qual_metrics_df()
# quan_metrics = pd.read_csv('/opt/ml/quan_metrics_df.csv') # 정량 지표들이 array로 담겨있음
# quan_metrics = pd.read_csv('/opt/ml/quan_metrics_df.csv') # 정량 지표들이 array로 담겨있음

# fig_total = plot_total_metrics(total_metrics)
fig_qual = plot_qual_metrics(qual_metrics)
fig_dist = plot_dist_for_metrics(qual_metrics, 'Diversity(jaccard)')

### layout 정의
#### side bar : 비교하고 싶은 실험 추가하고 삭제하는 부분
sidebar = html.Div([
        html.H3("Select Expriements",),
        html.Hr(),
        html.Div(id='model_form', children=[]),
        
        dbc.Button('➕', id='add_button', n_clicks=0, style={'position':'absolute', 'right':0, 'margin-right':'2rem'}),
        dbc.Popover("Add a new expriement", trigger='hover', target='add_button', body=True),
        dbc.Button('Compare!', id='compare_btn', n_clicks=0)
    ],
    className='sidebar'
)

#### total metric 그래프 그릴 부분
total_graph = html.Div([
    html.Br(),
    html.H1(children='Model vs Model', style={'text-align': 'center','font-weight': 'bold'}),
    html.Hr(),
    
    html.Div(id='select_model'),
    html.Div(id='select_model2'),
    

    html.H3('Total Metric'),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Br(),
            ]),
            html.Div(id='total_metric')
            # dcc.Graph(id='total_metric') # html.Div(id='total_metric')
        ]),
    ])
])

#### 정량, 정성 지표 그래프 그릴 부분
specific_metric = html.Div([
    html.H3('Specific Metric'),
    dbc.Row([
        dbc.Col([
            dbc.RadioItems(
                id="sort_of_metric",
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=[
                    {"label": "Qualitive", "value": 'Qual'},
                    {"label": "Quantitive", "value": 'Quant'},
                ],
                value='Qual',
            ),
            html.Br(),
            dcc.Dropdown(id='metric_list')
            ], width=4),
        dbc.Col([
            dcc.Graph(id = 'bar_fig'), #figure=fig_qual
            html.Div(id = 'dist_fig'),  # dcc.Graph(id = 'dist_fig')
        ], width=8)
    ]),
    ],
    className="radio-group", 
)


layout = html.Div(children=[
    gct.get_navbar(has_sidebar=False),
    html.Div([
    sidebar,
    total_graph,
    specific_metric]),
    dcc.Store(id='store_selected_exp', storage_type='session'),
    dcc.Store(id='store_exp_names', storage_type='session'),
    dcc.Store(id='store_exp_ids', storage_type='session')
    

], className="content")


### 어떤 실험을 고를지 select하는 dropdown을 보여주는 callback
@callback(
    Output("model_form", "children"),
    [
        Input("add_button", "n_clicks"),
        Input({"type": "delete_btn", "index": ALL}, "n_clicks"),
        Input("store_exp_names", "data")
    ],
    [State("model_form", "children")],
)
def display_dropdowns(n_clicks, _, store_exp_names, children): 
    input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if "index" in input_id:
        delete_chart = json.loads(input_id)["index"]
        children = [
            chart
            for chart in children
            if "'index': " + str(delete_chart) not in str(chart)
        ]
    else:
        model_form = html.Div([
                dbc.Button('➖', className='delete-btn', id={'type':'delete_btn', 'index':n_clicks}),
                # dbc.Popover("Delete this experiment", trigger='hover', target={'type':'delete_btn', 'index':ALL}, body=True), # 동적 컴포넌트에는 어떻게 적용해야 할지 모르겠음
                dcc.Dropdown(
                    store_exp_names, id={'type':'selected_exp', 'index':n_clicks},
                    placeholder="Select or Search experiment", optionHeight=50, # options=[{'label':'exp_name', 'value':'exp_id'}, ... ], # label은 보여지는거, value는 실제 어떤 데이터인지
                ),
                html.Hr(),
                html.P(id={'type':"exp's_hype", 'index':n_clicks}),
                html.Br()
            ], className='form-style')
        children.append(model_form)
    return children

### exp_ids가 들어오면 실험 정보들 return 하는 callback
@callback(
    Output('selected_model_info', 'data'),
    Input('store_exp_ids', 'data')
)
def get_stored_selected_models(exp_ids:list[int]) -> pd.DataFrame:
    params = {'ID':'mkdir', 'dataset_name':'ml-1m', 'exp_id': exp_ids}
    response = requests.get(API_url + '/frontend/selected_models', params = params)
    a = response.json()
    return pd.DataFrame.from_dict(data=a, orinet='tight')

### selected_exp 의 hype을 소개하는 callback
@callback(
    Output({"type": "exp's_hype", "index": MATCH}, "children"),
    [
        Input({"type": "selected_exp", "index": MATCH}, "value"),
        Input('selected_model_info', 'data')
    ],
)
def display_output(selected_exp:str, df:pd.DataFrame) -> str:
    exp_hype = df.loc[df['experiment_name'] == selected_exp, 'hyperparameters']
    return exp_hype.values() 

# TODO: get request(selected user) and plot total_metrics
# def get_quantative_metrics(form): 
#     params={'model_name': form['model'], 'str_key': form['values']}
#     return requests.get(url=f'{API_url}/metric/quantitative/', params=params).json()[0]


### compare! 버튼을 누르면 plot을 그려주는 callback
@callback(  # compare 버튼 누름
        Output('total_metric', 'children'),
        Input('compare_btn', 'n_clicks'),
        State('compare_btn', 'n_clicks'),
        # prevent_initial_call=True
)
def plot_total_metrics(inp, state): # df:pd.DataFrame
    if state == 0:
        return html.Div([
            html.P("If you want to metric compare between selected models, Click Compare!"),
        ])
    else:
        # 모델간 정량, 정성 지표 plot (Compare Table에 있는 모든 정보들 활용)
        metrics = list(total_metrics.columns[1:])
        colors = ['#A56CC1', '#A6ACEC', '#63F5EF', '#425FEF'] # 사용자 입력으로 받을 수 있어야 함
        
        fig = go.Figure()

        for i in total_metrics.index:
            fig.add_bar(name=total_metrics['Name'][i], x=metrics, y=list(total_metrics.iloc[i,1:].apply(np.mean)), text=list(total_metrics.iloc[i,1:].apply(np.mean)), marker={'color' : colors[i]})
            # .apply(eval)은 np.array나 list를 문자열로 인식할 때만 활용해주면 됨
            # 아니면 TypeError: eval() arg 1 must be a string, bytes or code object 발생
        fig.update_layout(
            barmode='group',
            bargap=0.15, # gap between bars of adjacent location coordinates.
            bargroupgap=0.1, # gap between bars of the same location coordinate.)
            title_text='Metric indicators'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

        return dcc.Graph(figure=fig)  # id = 'total_metric'
    # return fig


### ??
# @callback(
#         Output('map', 'children'),
#         Input('compare_btn', 'n_clicks'),
#         prevent_initial_call=True
# )
# def get_quantative_metrics(form):
#     params={'model_name': form['model'], 'str_key': form['values']}
#     return requests.get(url=f'{gct.API_URL}/metric/quantitative/', params=params).json()[0]

### ??
# @callback(
#     Output('select_model2', 'children'),
#     Input('uname-box', 'value'),
#     prevent_initial_call=True
# )
# def sider_custom_trigger_demo(v):

#     return v




### metric lists를 보여주는 callback
@callback(
    Output('metric_list', 'options'),
    Input('sort_of_metric', 'value'),
)
def load_metric_list(sort_of_metric:str) -> list:
    if sort_of_metric == 'Quant':
        metric_list = ['Recall_k', 'NDCG', 'AP@K', 'AvgPopularity', 'TailPercentage']
    elif sort_of_metric == 'Qual':
        metric_list = ['Diversity(jaccard)', 'Diversity(cosine)', 'Serendipity(jaccard)', 'Serendipity(PMI)', 'Novelty']
    return metric_list

### Qual or Quant 선택하면 metric bar plot 띄워주는 Callback
@callback(
    Output('bar_fig', 'figure'),
    Input("sort_of_metric", 'value'),
)
def plot_bar(sort_of_metric):
    if sort_of_metric == 'Qual':
        fig = plot_qual_metrics(qual_metrics)
        return fig
    # elif sort_of_metric == 'Quant':
    #     fig = plot_quant_metrics(quant_metrics)
    else:
        return html.Div([])
    

### 선택한 metric에 대한 dist plot을 띄워주는 callback
@callback(
    Output('dist_fig', 'children'),
    Input("metric_list", 'value'),
)
def plot_dist(value):
    if value in ['Diversity(jaccard)', 'Diversity(cosine)', 'Serendipity(jaccard)', 'Serendipity(PMI)', 'Novelty']:
        fig = plot_dist_for_metrics(qual_metrics, value)
        return dcc.Graph(id = 'dist_fig', figure=fig)
    # elif value in ['Recall_k', 'NDCG', 'AP@K', 'AvgPopularity', 'TailPercentage']:
        # fig = plot_dist_for_metrics(quan_metrics, value)    
        # return dcc.Graph(id = 'dist_fig', figure=fig)
    else:
        return html.Div([])
    

# ### Quant 지표 선택시 그림 그려지는 callback
# @callback(
#     Output('Quant_fig', 'figure'),
#     Input('metric_list', 'options')
# )
# def plot_Quant_dist():
#     fig = go.Figure()
#     return fig
