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

# user_df = pd.read_csv('/opt/ml/user.csv', index_col='user_id')[:100]

dash.register_page(__name__, path='/compare-table')


pinned_column_name = 'experiment_name'
pinned_column_setting = dict(
                        pinned='left',
                        checkboxSelection=True,
                        )


select_dataset = html.Div([
    html.H3('데이터 선택'),
    dcc.Dropdown(
                 id='dataset-list',
                 className="mt-3 w-25"
                 ),
    html.Hr()
])

def get_table(df):
    compare_table = html.Div([
        html.Div([
        html.H3(['전체 실험 목록', html.Span(" �", id="compare-table-tooltip")], className="mb-3s"),
        dbc.Button('선택 완료', id='select_done', n_clicks=0, color="success"),
        html.Div(id="guide-to-model-vs"),
        ], className="hstack gap-5 mb-3"),
        dbc.Tooltip("비교할 실험을 선택하고 "
                    "'선택 완료' 버튼을 "
                    "눌러보세요!",
                     target="compare-table-tooltip",
                     className="w-auto"),
        dbc.Row([
            dbc.Col([
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
                ]),

                html.Div(id="selected_table_container", className="mt-0 hstack gap-3")

        ]),
        html.Br(),
        # dbc.Button('Select done!', id='select_done', n_clicks=0),
        # html.Hr(),

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

        # html.Div(id="selected_table_container"),
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
        Input("store_user_state", "data"),
        State("store_user_state", "data")
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
        Output('store_user_dataset', 'data'),
        State("store_user_state", "data"),
        Input("dataset-list", "value"),
        prevent_initial_call=True
)
def get_exp_data(user_state:dict, dataset_name:str,):
    if dataset_name == None:
        return dbc.Alert("데이터셋을 먼저 선택해주세요.", color="info", className="w-50"), None, None
    params = {
        "ID": user_state["username"],
        "dataset_name": dataset_name
    }
    response = requests.get(f"{gct.API_URL}/frontend/get_exp_total", params=params)
    df = pd.DataFrame.from_dict(response.json(), orient="tight")
    # 컬럼 순서 변경
    temp_col1 = df.columns[4:].to_list()
    temp_col2 = df.columns[:4].to_list()
    df = df[temp_col1+temp_col2]
    return get_table(df), df.columns, dataset_name

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
    # AgGrid = dag.AgGrid(
    #     id = 'selected_table',
    #     rowData=seleceted_rows,
    #     columnDefs=[
    #          {'headerName': column, 'field':column, 'pinned':'left', 'checkboxSelection':True, 'rowDrag':True, 'headerCheckboxSelection':True} if column == pinned_column_name else {'headerName': column, 'field':column, } for column in exp_column
    #     ],
    #     # columnSize="sizeToFit",
    #     defaultColDef=dict(
    #             resizable= True,
    #             sortable=True,
    #             filter=True,
    #             floatingFilter=True,
    #             headerCheckboxSelectionFilteredOnly=True,
    #     ),
    #     dashGridOptions=dict(
    #         rowSelection="multiple",
    #         ),
    #     rowDragManaged=True,
    #     animateRows=True,
    # )
    selects = [html.H6("선택한 실험 목록: ", className="mt-3 ")]
    if seleceted_rows == None:
        PreventUpdate
    else:
        for row in seleceted_rows:
            selects.append(
                dbc.Badge(row["experiment_name"], color="info", className="mt-3")
                )
    return selects, seleceted_rows

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

@callback(
    Output("guide-to-model-vs", "children"),
    Input('select_done', 'n_clicks'),
    prevent_initial_update=True
)
def guide_to_model_vs(n):
    if n == 0:
        return None
    return dbc.Alert("Model vs Model 페이지로 이동해보세요!", color="info", className="mb-0 mt-0"),
