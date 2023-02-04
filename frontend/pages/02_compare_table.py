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


pinned_column_name = 'experiment_name'
pinned_column_setting = dict(
                        pinned='left',
                        checkboxSelection=True,
                        )


select_dataset = html.Div([
    html.H3('Select a dataset'),
    dcc.Dropdown(
                 id='dataset-list',
                 style={'width':'40%'},
                 ),
    html.Hr()
])

def get_table(df):
    compare_table = html.Div([
        html.H3('Total experiments'),
        dag.AgGrid(
            rowData=df.to_dict('records'),
            id='compare_table',
            columnDefs=[
                {'headerName': column, 'field':column, 'pinned':'left', 'checkboxSelection':True, 'rowDrag':True, 'headerCheckboxSelection':True} if column == pinned_column_name else {'headerName': column, 'field':column, } for column in df.columns 
            ],
            # columnSize="sizeToFit",
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
        dbc.Button('Select done!', id='select_done', n_clicks=0),
        html.Hr(),
        
    ], )
    return compare_table

layout = html.Div([
    gct.get_navbar(has_sidebar=False),
    html.P(id='test_data'),
    html.Div(id="test2"),
    html.Div([
        select_dataset,
        html.Div(id="exp_table_container"),
        # compare_table,
        
        html.Div(id="selected_table_container"),
        # selected_table,
        html.H3(id='output_test')
    ],
    className="container"),

    dcc.Store(id='store_selected_exp', storage_type='session'),
    dcc.Store(id='store_exp_names', storage_type='session'),
    dcc.Store(id='store_exp_ids', storage_type='session'),
    dcc.Store(id='store_exp_column', storage_type='session')
])

@callback(
        Output('dataset-list', 'options'),
        Input("user_state", "data"),
        State("user_state", "data")
)
def test_request(n, user_state):
    params = {
        "ID": user_state["username"]
    }
    response = requests.get(f"{gct.API_URL}/web4rec-lib/check_dataset", params=params)
    if response.status_code == 201:
        return response.json()
    
@callback(
        Output('exp_table_container', 'children'),
        Output('store_exp_column','data'),
        State("user_state", "data"),
        Input("dataset-list", "value"),
        prevent_initial_call=True
)
def get_exp_data(user_state:dict, dataset_name:str,):
    if dataset_name == None:
        return dbc.Alert("데이터셋을 먼저 선택해주세요.", color="primary"), None
    params = {
        "ID": user_state["username"],
        "dataset_name": dataset_name
    }
    response = requests.get(f"{gct.API_URL}/frontend/get_exp_total", params=params)
    df = pd.DataFrame.from_dict(response.json(), orient="tight")
    return get_table(df), df.columns

## 선택한 실험의 정보를 table로 만들어주고, 그 실험 정보 자체를 return
@callback(  
    Output('selected_table_container', 'children'),
    Output('store_selected_exp', 'data'),
    Input('select_done', 'n_clicks'),
    State('compare_table', 'selectionChanged'),
    State('store_exp_column', 'data'),
    prevent_initial_call=True
)
def plot_selected_table(n, seleceted_rows, exp_column):
    AgGrid = dag.AgGrid(
        id = 'selected_table',
        rowData=seleceted_rows,
        columnDefs=[
             {'headerName': column, 'field':column, 'pinned':'left', 'checkboxSelection':True, 'rowDrag':True, 'headerCheckboxSelection':True} if column == pinned_column_name else {'headerName': column, 'field':column, } for column in exp_column
        ],
        # columnSize="sizeToFit",
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
    
    return [html.H3('Selected experiments'),
    html.Div(AgGrid, id='table-container')], seleceted_rows

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
        exp_ids.append(each['exp_id'])
    return exp_ids