import dash
from dash import html, dcc, callback, Input, Output, State,  MATCH, ALL, dash_table
import dash_bootstrap_components as dbc
import requests
import pandas as pd
from dash.exceptions import PreventUpdate
from . import global_component as gct
import json


API_url = 'http://127.0.0.1:8000'

user_df = pd.read_csv('/opt/ml/user.csv', index_col='user_id')[:100]

dash.register_page(__name__, path='/compare-table')

compare_table = html.Div([
    dash_table.DataTable(
        data=user_df.to_dict('records'),
        id='compare_table',
        columns=[
            {'name': column, 'id':column} for column in user_df.columns
        ],
        filter_action='native',
        sort_action='native',
        row_selectable="multi",
        selected_rows=[],
        page_action='native',
        page_current=0,
        page_size=10,
        fixed_columns={'headers': True, 'data': 1},
        style_table={'minWidth':'100%'},
        # style_cell={
        #     'overflow': 'hidden',
        # }
    )
])

selected_table = html.Div(children=[],id='selected_table',)

layout = html.Div([
    gct.noside_navbar,
    compare_table,
    selected_table
])

@callback(
    Output('selected_table', 'children'),
    Input('compare_table', 'selected_rows'),
    # State('compare_table', 'selected_row'),
    # State('selected_table', 'children')
)
def select(selected_row, ) -> str:
    
    return str(selected_row)

@callback(
    Output('selected_table', 'style_data_conditional'),
    Input('compare_table', 'selected_row_ids')
)
def update_styles(selected_rows):
    return [{
        'if': { 'row_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_rows]

    colors = ['#FF69B4' if id == active_row_id
            else '#7FDBFF' if id in selected_id_set
            else '#0074D9'
            for id in row_ids]
