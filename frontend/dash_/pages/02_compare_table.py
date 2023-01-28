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
                        # rowDrag=True,
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
    html.H3('Selected experiments'),
    dag.AgGrid(
        id = 'selected_table',
        rowData=[],
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
])

selected_table = html.Div(children=[],id='selected_table',)

layout = html.Div([
    gct.get_navbar(has_sidebar=False),
    compare_table,
    selected_table,
    html.H3( id='output_test')
])

@callback(
    Output('selected_table', 'rowData'),
    # Output('output_test', 'children'),
    Input('compare_table', 'selectionChanged'),
    State('compare_table', 'selectionChanged'),
    prevent_initial_call=True
)
def test_output(r, r2):
    r2.append(r)
    # return str(r2) + str(user_df.to_dict('records'))
    return r2

# @callback(
#     Output('selected_table', 'children'),
#     Input('compare_table', 'selected_rows'),
#     # State('compare_table', 'selected_row'),
#     # State('selected_table', 'children')
# )
# def select(selected_row, ) -> str:
    
#     return str(selected_row)

# @callback(
#     Output('selected_table', 'style_data_conditional'),
#     Input('compare_table', 'selected_row_ids')
# )
# def update_styles(selected_rows):
#     return [{
#         'if': { 'row_id': i },
#         'background_color': '#D2F3FF'
#     } for i in selected_rows]

#     colors = ['#FF69B4' if id == active_row_id
#             else '#7FDBFF' if id in selected_id_set
#             else '#0074D9'
#             for id in row_ids]

# @callback(
#     Output('datatable-interactivity', 'style_data_conditional'),
#     Input('compare_table', 'selected_columns'),
#     State('compare_table', 'selected_columns')
# )
# def update_styles(selected_columns, now_selected) -> dict :
    
#     now_selected.update()

#     return now_selected