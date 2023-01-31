import dash
from dash import html, dcc, callback, Input, Output, State,  MATCH, ALL
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
import feffery_antd_components as fac
from . import global_component as gct
import json

API_url = 'http://127.0.0.1:8000'

dash.register_page(__name__, path='/model-vs-model')

exp_df = pd.DataFrame(columns = ['model','recall','ndcg','map','popularity','colors'])

exp_df.loc[1,:] = ['M1',0.1084,0.0847,0.1011,0.0527,'red']
exp_df.loc[2,:] = ['M2',0.1124,0.0777,0.1217,0.0781,'green']
exp_df.loc[3,:] = ['M3',0.1515,0.1022,0.1195,0.0999,'blue']
exp_df.loc[4,:] = ['M4',0.0917,0.0698,0.0987,0.0315,'goldenrod']



fig_total = px.bar(
            exp_df,
            x = 'model',
            y ='recall',
            color = 'model',
            color_discrete_sequence=exp_df['colors'].values
            )

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
        dbc.Button('Compare!')
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
          dcc.Graph(figure=fig_total, id='total_metric')
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
            dcc.Graph(figure=fig_total),
            dcc.Graph(figure=fig_total),
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


@callback(
        Output('map', 'children'),
        Input('compare_btn', 'n_clicks'),
        prevent_initial_call=True
)
def get_quantative_metrics(form):
    params={'model_name': form['model'], 'str_key': form['values']}
    return requests.get(url=f'{API_url}/metric/quantitative/', params=params).json()[0]

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