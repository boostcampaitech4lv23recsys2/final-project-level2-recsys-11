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

total_metrics = None
total_metrics_users = None

sidebar = html.Div([
        html.H3("실험 선택",),
        html.Hr(),
        html.Div(id='model_form', children=[]),

        html.Div([
            dbc.Button('➕', id='add_button', n_clicks=0, 
                    #    style={'position':'absolute', 'right':0, 'margin-right':'2rem'}, 
                    className="mt-1 me-5"),
            dbc.Popover("Add a new expriement", trigger='hover', target='add_button', body=True),
            dbc.Button('비교하기', id='compare_btn', n_clicks=0, 
                    className="ms-5 mt-1 w-50"
                    )
        ], className="hstack gap-5")
    ],
    className='sidebar'
)

#### total metric 그래프 그릴 부분
total_graph = html.Div([
    html.Div(id='select_model2'),
    html.Div(id='total_metric'),
    dbc.Row([
        dbc.Col([
            html.Div([
                
            ]),
            # html.Div(id='total_metric')
            # dcc.Graph(id='total_metric') # html.Div(id='total_metric')
        ]),
    ])
])

#### 정량, 정성 지표 그래프 그릴 부분
def specific_metric():
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
                        {"label": "Qualitative", "value": 'Qual'},
                        {"label": "Quantitative", "value": 'Quant'},
                    ],
                    value='Qual',
                ),
                html.Br(),
                dcc.Dropdown(id='metric_list')
                ], width=3),
                html.Br(),
                html.Div([html.P(id="print_metric"),]),
            dbc.Col([
                dcc.Graph(id = 'bar_fig'), #figure=fig_qual
                html.Div(id = 'dist_fig'),  # dcc.Graph(id = 'dist_fig')
            ], width=8)
        ]),

        ],
        className="radio-group",
    )
    return specific_metric


layout = html.Div([html.Div(
    [gct.get_navbar(has_sidebar=False),]),
    html.Div([
    sidebar,
    html.Br(),
    html.H1(children='Model vs Model', style={'text-align': 'center','font-weight': 'bold'}),
    html.Hr(),
    total_graph,
    html.Div(id = 'specific_metric_children')
    ], className="content m-"),
    html.Div(id='trash'),
    dcc.Store(id='store_selected_exp', storage_type='session'),
    dcc.Store(id='store_exp_names', storage_type='session'),
    dcc.Store(id='store_exp_ids', storage_type='session'),
    dcc.Store(id='store_selected_exp_names', data=[], storage_type='session')

])


### exp_ids가 들어오면 실험 정보들 return 하는 callback
@callback(
    Output('trash', 'children'),
    Input('compare_btn', 'n_clicks'),
    State('store_exp_ids', 'data')
)
def get_stored_selected_models(n, exp_ids:list[int]) -> pd.DataFrame:
    global total_metrics
    global total_metrics_users
    params = {'ID':'mkdir', 'dataset_name':'ml-1m', 'exp_ids': exp_ids}
    response = requests.get(API_url + '/frontend/selected_metrics', params = params)
    a = response.json()
    total_metrics = pd.DataFrame().from_dict(a['model_metrics'], orient='tight')
    total_metrics_users = pd.DataFrame().from_dict(a['user_metrics'], orient='tight')
    return html.Div([])

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
                dbc.Button('➖', className='mb-3', id={'type':'delete_btn', 'index':n_clicks}),
                # dbc.Popover("Delete this experiment", trigger='hover', target={'type':'delete_btn', 'index':ALL}, body=True), # 동적 컴포넌트에는 어떻게 적용해야 할지 모르겠음
                dcc.Dropdown(
                    store_exp_names, id={'type':'selected_exp', 'index':n_clicks},
                    placeholder="Select or Search experiment", optionHeight=50, # options=[{'label':'exp_name', 'value':'exp_id'}, ... ], # label은 보여지는거, value는 실제 어떤 데이터인지
                ),
                html.Hr(),
                dcc.Markdown(id={'type':"exp's_hype", 'index':n_clicks}, dangerously_allow_html=True),
                html.Br()
            ], className='form-style my-2')
        children.append(model_form)
    return children


### selected_exp 의 hype을 소개하는 callback
@callback(
    Output({"type": "exp's_hype", "index": MATCH}, "children"),
    [
        Input({"type": "selected_exp", "index": MATCH}, "value"),
        State('store_selected_exp', 'data')
    ],
)
def display_output(selected_dropdown:str, data) -> str: #
    if selected_dropdown is None:
        raise PreventUpdate
    
    tmp_df = pd.DataFrame(data).set_index('experiment_name')
    exp_hype = tmp_df.loc[selected_dropdown,'hyperparameters']
    exp_hype = exp_hype[1:-1]
    exp_hype = exp_hype.split(',')
    # TODO: 마크다운으로 리턴
    
    exp_hype = "<br>".join(exp_hype)
    return exp_hype

### selected_exp의 experiment_name을 저장
@callback(
    Output('store_selected_exp_names', 'data'),
    Input({"type": "selected_exp", "index": ALL}, 'value'),
    State('store_selected_exp_names', 'data')
)
def save_selected_exp_names(value, data):
    if value == [None]:
        raise PreventUpdate
    return value


### compare! 버튼을 누르면 plot을 그려주는 callback
@callback(  # compare 버튼 누름
        Output('specific_metric_children', 'children'),
        Output('total_metric', 'children'),
        Input('store_selected_exp_names', 'data'),
        Input('compare_btn', 'n_clicks'),
        State('compare_btn', 'n_clicks'),
        State('store_selected_exp','data')
        # prevent_initial_call=True
)
def plot_total_metrics(data, n, state, store): # df:pd.DataFrame
    if state == 0:
        return html.Div([]), dbc.Alert("왼쪽에서 모델을 선택하고 Compare 버튼을 눌러 실험들의 지표를 확인해보세요!", color="info", className="w-75")
        
    else:
        # 모델간 정량, 정성 지표 plot (Compare Table에 있는 모든 정보들 활용)
        colors = ['#9771D0', '#D47DB2', '#5C1F47', '#304591', '#BAE8C8', '#ECEBC6', '#3D3D3D'] # 사용자 입력으로 받을 수 있어야 함
        store_df = pd.DataFrame(store).set_index('experiment_name')
        tmp_metrics = total_metrics.drop(['diversity_jac','serendipity_jac'], axis=1)
        metrics = list(tmp_metrics.columns)
        fig = go.Figure()
        for i,exp_name in enumerate(data):
            exp_id = store_df.loc[exp_name, 'exp_id'] # exp_name에 맞는 exp_id 찾아주기
            fig.add_bar(name=exp_name, x=metrics, y=list(tmp_metrics.loc[exp_id,:]), text=list(tmp_metrics.loc[exp_id,:]), marker={'color' : colors[i]})
            # .apply(eval)은 np.array나 list를 문자열로 인식할 때만 활용해주면 됨
            # 아니면 TypeError: eval() arg 1 must be a string, bytes or code object 발생
        fig.update_layout(
            barmode='group',
            bargap=0.15, # gap between bars of adjacent location coordinates.
            bargroupgap=0.1, # gap between bars of the same location coordinate.)
            title_text='Metric indicators'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

        return specific_metric(), [html.H3('전체 지표 정보'),dcc.Graph(figure=fig)]  # id = 'total_metric'


### metric lists를 보여주는 callback
@callback(
    Output('metric_list', 'options'),
    Input('sort_of_metric', 'value'),
)
def load_metric_list(sort_of_metric:str) -> list:
    if sort_of_metric == 'Quant':
        metric_list = [
            {'label': 'Recall_k', 'value' : 'recall'},
            {'label':'NDCG', 'value':'ndcg'},
            {'label':'AP@K', 'value':'map'},
            {'label':'AvgPopularity', 'value':'avg_popularity'},
            {'label':'TailPercentage', 'value':'tail_percentage'}
            ]
    elif sort_of_metric == 'Qual':
        metric_list = [
            {'label':'Diversity(jaccard)', 'value':'diversity_jac'},
            {'label':'Diversity(cosine)', 'value':'diversity_cos'},
            {'label':'Serendipity(jaccard)', 'value':'serendipity_jac'},
            {'label':'Serendipity(PMI)', 'value':'serendipity_pmi'},
            {'label':'Novelty', 'value':'novelty'},
            ]
    return metric_list

### Qual or Quant 선택하면 metric bar plot 띄워주는 Callback
@callback(
    Output('bar_fig', 'figure'),
    State('store_selected_exp_names', 'data'),
    Input("sort_of_metric", 'value'),
    State('store_selected_exp','data')
)
def plot_bar(data, sort_of_metric, store):
    store_df = pd.DataFrame(store).set_index('experiment_name')
    colors = ['#9771D0', '#D47DB2', '#5C1F47', '#304591', '#BAE8C8', '#ECEBC6', '#3D3D3D']
    if sort_of_metric == 'Qual':
        qual_metrics = total_metrics.iloc[:,6:]
        metrics = list(qual_metrics.columns)

        fig = go.Figure()
        for i,exp_name in enumerate(data):
            exp_id = store_df.loc[exp_name, 'exp_id'] # exp_name에 맞는 exp_id 찾아주기
            fig.add_bar(name=exp_name, x=metrics, y=list(qual_metrics.loc[exp_id,:]), text=list(qual_metrics.loc[exp_id,:]), marker={'color' : colors[i]})

        fig.update_layout(
            barmode='group',
            bargap=0.15, # gap between bars of adjacent location coordinates.
            bargroupgap=0.1, # gap between bars of the same location coordinate.)
            title_text='Specific Qualitative Metrics'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        return fig

    elif sort_of_metric == 'Quant':
        quant_metrics = total_metrics.iloc[:,:6]
        metrics = list(quant_metrics.columns)

        fig = go.Figure()
        for i,exp_name in enumerate(data):
            exp_id = store_df.loc[exp_name, 'exp_id'] # exp_name에 맞는 exp_id 찾아주기
            fig.add_bar(name=exp_name, x=metrics, y=list(quant_metrics.loc[exp_id,:]), text=list(quant_metrics.loc[exp_id,:]), marker={'color' : colors[i]})

        fig.update_layout(
            barmode='group',
            bargap=0.15, # gap between bars of adjacent location coordinates.
            bargroupgap=0.1, # gap between bars of the same location coordinate.)
            title_text='Quantitative indicators'
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
    Output('dist_fig', 'children'),
    State('store_selected_exp_names', 'data'),
    Input("metric_list", 'value'),
)
def plot_dist(data, value):
    colors = ['#9771D0', '#D47DB2', '#5C1F47', '#304591', '#BAE8C8', '#ECEBC6', '#3D3D3D']
    if value in ['diversity_jac', 'diversity_cos', 'serendipity_pmi', 'serendipity_jac', 'novelty']:
        group_labels = data
        colors = colors[:len(data)]
        hist_data = total_metrics_users[value].values
        fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                                bin_size=0.025, show_rug=True, curve_type='kde')

        fig.update_layout(title_text=f'Distribution of {value}')
        return dcc.Graph(figure=fig)

    elif value in ['recall', 'ndcg', 'map', 'avg_popularity', 'tail_percentage']:
        if value == 'map':
            value = 'avg_precision'
        group_labels = data
        colors = colors[:len(data)]
        hist_data = total_metrics_users[value].values
        fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                                bin_size=0.025, show_rug=True, curve_type='kde')
        fig.update_layout(title_text=f'Distribution of {value}')
        return dcc.Graph(figure=fig)
    else:
        return html.Div([])
