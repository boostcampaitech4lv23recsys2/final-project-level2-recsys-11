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

API_url = 'http://127.0.0.1:8000'

user_df = pd.read_csv('/opt/ml/user.csv', index_col='user_id')[:100]

dash.register_page(__name__, path='/compare-table')

pinned_column_name = 'age'
pinned_column_setting = dict(
                        pinned='left',
                        checkboxSelection=True,
                        )


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
])

selected_table = html.Div(children=[],id='selected_table',)

layout = html.Div([
    gct.get_navbar(has_sidebar=False),
    compare_table,
    selected_table,
    html.H3( id='output_test')
])

@callback(
    Output('table-container', 'children'),
    Input('select_done', 'n_clicks'),
    State('compare_table', 'selectionChanged'),
    prevent_initial_call=True
)
def test_output(n, r2):

    return dag.AgGrid(
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