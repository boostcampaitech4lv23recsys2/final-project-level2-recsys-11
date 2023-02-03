import dash
from dash import html, dcc, callback, Input, Output, State,  MATCH, ALL, dash_table
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import requests
import pandas as pd
from dash.exceptions import PreventUpdate
from . import global_component as gct
import json
# AgGrid docs:
# https://www.ag-grid.com/javascript-data-grid/column-pinning/

user_df = pd.read_csv('/opt/ml/user.csv', index_col='user_id')[:100]

dash.register_page(__name__, path='/compare-table')

# TODO: dataset_list 넘겨주는 API 필요
# dataset_list = requests.get('/dataset_list', params={'user_id': 'smth'})

pinned_column_name = 'age'
pinned_column_setting = dict(
                        pinned='left',
                        checkboxSelection=True,
                        )

select_dataset = html.Div([
    html.H3('Select a dataset'),
    dcc.Dropdown(
        # TODO: dataset_list 넘겨주는 API 필요
        # dataset_list,
        # dataset_list[0],
                 id='dataset-list',
                 style={'width':'40%'},
                 ),
    html.Hr()
])

compare_table = html.Div([
    html.H3('Total experiments'),
    dag.AgGrid(
        rowData=user_df.to_dict('records'),
        id='compare_table',
        columnDefs=[
             {'headerName': column, 'field':column, 'pinned':'left', 'checkboxSelection':True, 'rowDrag':True, 'headerCheckboxSelection':True} if column == pinned_column_name else {'headerName': column, 'field':column, } for column in user_df.columns 
        ],
        columnSize="sizeToFit",
        defaultColDef=dict(
                resizable= True,
                sortable=True,
                filter=True,
                floatingFilter=True,
                headerCheckboxSelectionFilteredOnly=True,
        ),
        dashGridOptions=dict(
            rowSelection="multiple", 
            ),
        rowDragManaged=True,
        animateRows=True,
    ),
    html.Br(),
    dbc.Button('Select done!', id='select_done'),
    html.Hr(),
    html.H3('Selected experiments'),
    html.Div(id='table-container'),
], )

selected_table = html.Div(children=[], id='selected_table')
# store_data = html.Div(children=[], id='store_data')

layout = html.Div([
    gct.get_navbar(has_sidebar=False),
    html.Div(id='test_store'),
    html.Div([
        select_dataset,
        compare_table,
        selected_table,
        html.H3(id='output_test')
    ],
    className="container"),
    html.Div(id='store_exp_names'),

    dcc.Store(id='store_selected_exp', storage_type='memory'),
    dcc.Store(id='store_exp_names', storage_type='memory'),
    dcc.Store(id='store_exp_ids', storage_type='memory')
])
 
# @callback(
#     Output('dataset-list', 'options'),
#     Input('select_done', 'n_clicks'),
#     State('user_state', 'data')
# )
# def get_dataset_list(n, user_state):

#     response = requests.post(f"{gct.API_URL}/user/get_current_user", json=user_state)
#     if response.status_code == 201:
#         return [1,2,3,4]
#     else:
#         return list(str(response))

# gct.Authenticate('select_done', 'datase_list')

    # response = requests.post(f"{gct.API_URL}/user/get_current_user", json=user_state)
    # if response.status_code == 201:
    #     return [1,2,3,4]
    # else:
    #     return list(str(response))
    
# @callback(
#     Output('test_store', 'children'),
#     Input('select_done', 'n_clicks'),
#     State('user_state', 'data'),
#     prevent_initial_call=True
# )
# def test_store(n, data):
#     return
# @callback(
#     Output('test_store', 'children'),
#     Input('select_done', 'n_clicks'),
#     State('user_state', 'data'),
#     prevent_initial_call=True
# )
# def test_store(n_clicks, data):
#     return str(data)

## 선택한 실험의 정보를 table로 만들어주고, 그 실험 정보 자체를 return
@callback(  
    Output('table-container', 'children'),
    Output('store_selected_exp', 'data'),
    Input('select_done', 'n_clicks'),
    State('compare_table', 'selectionChanged'),
    prevent_initial_call=True
)
def test_output(n, r2):
    AgGrid = dag.AgGrid(
        id = 'selected_table',
        rowData=r2,
        columnDefs=[
             {'headerName': column, 'field':column, 'pinned':'left', 'checkboxSelection':True, 'rowDrag':True, 'headerCheckboxSelection':True} if column == pinned_column_name else {'headerName': column, 'field':column, } for column in user_df.columns 
        ],
        columnSize="sizeToFit",
        defaultColDef=dict(
                resizable= True,
                sortable=True,
                filter=True,
                floatingFilter=True,
                headerCheckboxSelectionFilteredOnly=True,
        ),
        dashGridOptions=dict(
            rowSelection="multiple", 
            ),
        rowDragManaged=True,
        animateRows=True,
    )

    return AgGrid, r2

## 선택한 실험에서 실험의 이름을 가져와서 model vs model page로 넘겨주기 (지금은 age로 임시방편)
@callback(
    Output('store_exp_names', 'data'),
    Input('store_selected_exp', 'data')
)
def store_selected_exp_names(data):
    if data is None:
        raise PreventUpdate
    exp_names = []
    for each in data:
        exp_names.append(each[pinned_column_name])
    return exp_names

## 선택한 실험에서 실험의 id를 가져와서 model vs model page로 넘겨주기
@callback(
    Output('store_exp_ids', 'data'),
    Input('store_selected_exp', 'data')
)
def store_selected_exp_ids(data):
    if data is None:
        raise PreventUpdate
    exp_ids = []
    for each in data:
        exp_ids.append(each[pinned_column_name])
    return exp_ids