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

API_url = 'http://127.0.0.1:8000'

dash.register_page(__name__, path='/model-vs-model')




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
    hist_data = [eval(each)[0] for each in qual_df['Diversity(jaccard)']]
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
qual_metrics = pd.read_csv('/opt/ml/qual_metrics_df.csv')

# fig_total = plot_total_metrics(total_metrics)
fig_qual = plot_qual_metrics(qual_metrics)
fig_dist = plot_dist_for_metrics(qual_metrics, 'Diversity(jaccard)')

model_form = html.Div([html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Button('➖', className='delete-btn', id='delete_button'),
            dbc.Popover("Delete this experiment", trigger='hover', target='delete_button', body=True)
]
            ),
]),
    dcc.Dropdown([1,2,3]),
    html.Hr(),
    html.P(
        f'''
        neg_sample: 123
        '''
    ),
], className='form-style'),
                       html.Br()])


sidebar = html.Div(
    [
        html.H3("Select Expriements",),
        html.Hr(),
        html.Div(id='model_form', children=[]),
        
        dbc.Button('➕', id='add_button', n_clicks=0, style={'position':'absolute', 'right':0, 'margin-right':'2rem'}),
        dbc.Popover("Add a new expriement", trigger='hover', target='add_button', body=True),
        dbc.Button('Compare!', id='compare_btn')
    ],
    className='sidebar'
)

total_graph = html.Div([
    html.Br(),
    html.H1(children='Model vs Model', style={'text-align': 'center','font-weight': 'bold'}),
    html.Hr(),
    
    html.Div(id='select_model'),
    html.Div(id='select_model2'),
    

    html.H3('Total Metric'),
    dbc.Row([
      dbc.Col([
          dcc.Graph(id='total_metric') # figure=fig_total,
            ]),
            ])
    ])

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
            dcc.Dropdown(options=['123', '12342'], id='metric_list')
            ], width=2),
        dbc.Col([
            dcc.Graph(figure=fig_qual),
            dcc.Graph(figure=fig_dist),
        ], width=8)
    ]),

           
    ],
    className="radio-group", 
)


layout = html.Div(children=[
    gct.get_navbar(),
    sidebar,
    total_graph,
    specific_metric
],className='content')


@callback(  # compare 버튼 누름
        Output('total_metric', 'figure'),
        Input('compare_btn', 'n_clicks'),
        prevent_initial_call=True
)
def plot_total_metrics(tmp): # df:pd.DataFrame
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

    return fig
# TODO: get request(selected user) and plot total_metrics
# def get_quantative_metrics(form): 
#     params={'model_name': form['model'], 'str_key': form['values']}
#     return requests.get(url=f'{API_url}/metric/quantitative/', params=params).json()[0]

@callback(
    Output('select_model2', 'children'),
    Input('uname-box', 'value'),
    prevent_initial_call=True
)
def sider_custom_trigger_demo(v):

    return v


@callback(
    Output("model_form", "children"),
    [
        Input("add_button", "n_clicks"),
        Input({"type": "delete_btn", "index": ALL}, "n_clicks"),
    ],
    [State("model_form", "children"), 
     ],
)
def display_dropdowns(n_clicks, _, children):
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
            dcc.Dropdown([1,2,3], value=1, id={'type':'selected_exp', 'index':n_clicks}),
            html.Hr(),
            html.P(
                    f'''
                    neg_sample: 123
                    '''
                , id={'type':"exp's_hype", 'index':n_clicks}),
            html.Br()
], className='form-style')
        children.append(model_form)
    return children

@callback(
    Output({"type": "exp's_hype", "index": MATCH}, "children"),
    [
        Input({"type": "selected_exp", "index": MATCH}, "value"),
    ],
)
def display_output(selected_exp:str) -> str:
    return f'{selected_exp}"s hype '

@callback(
    Output('metric_list', 'options'),
    Input('sort_of_metric', 'value'),
)
def load_metric_list(sort_of_metric:str) -> list:
    
    metric_list = ['fdsa', '123','fdsavcx']
    return metric_list