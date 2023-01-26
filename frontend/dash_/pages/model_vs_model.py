import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate
import feffery_antd_components as fac

API_url = 'http://127.0.0.1:8000'

dash.register_page(__name__)
load_figure_template("darkly") # figure 스타일 변경

model_hype_type = requests.get(url=f'{API_url}/model_hype_type').json()

exp_df = pd.DataFrame(columns = ['model','recall','ndcg','map','popularity','colors'])

exp_df.loc[1,:] = ['M1',0.1084,0.0847,0.1011,0.0527,'red']
exp_df.loc[2,:] = ['M2',0.1124,0.0777,0.1217,0.0781,'green']
exp_df.loc[3,:] = ['M3',0.1515,0.1022,0.1195,0.0999,'blue']
exp_df.loc[4,:] = ['M4',0.0917,0.0698,0.0987,0.0315,'goldenrod']

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

fig_recall = px.bar(
            exp_df,
            x = 'model',
            y ='recall',
            color = 'model',
            color_discrete_sequence=exp_df['colors'].values
            )
fig_ndcg = px.bar(
            exp_df,
            x = 'model',
            y ='ndcg',
            color = 'model',
            color_discrete_sequence=exp_df['colors'].values
            )
fig_map = px.bar(
            exp_df,
            x = 'model',
            y ='map',
            color = 'model',
            color_discrete_sequence=exp_df['colors'].values
            )
fig_popularity = px.bar(
            exp_df,
            x = 'model',
            y ='popularity',
            color = 'model',
            color_discrete_sequence=exp_df['colors'].values
            )
sidebar = html.Div([
    fac.AntdLayout([
        fac.AntdSider(
            [
                html.Div(
                    fac.AntdMenu(
                        menuItems=[
                            {
                                'component': 'Item',
                                'props':{
                                    'key': f'fdjsal',
                                    'title': f'fdjsal',
                                    
                                }
                            }
                        ],
                        mode='inline'
                    )
                )
            ], 
            collapsible=True,
            style={
                        'backgroundColor': 'rgb(240, 242, 245)',
                        'overflowY': 'auto'
                    }
            )
    ])
])

sidebar_2 = fac.AntdSpin(
    html.Div(
        [
            fac.AntdLayout(
                [
                    fac.AntdSider(
                        id='sider-custom-trigger-demo',
                        collapsible=True,
                        trigger=None,
                        style={
                            'backgroundColor': 'rgb(240, 242, 245)'
                        }
                    ),

                    fac.AntdContent(
                        fac.AntdButton(
                            '❌',
                            id='sider-custom-trigger-button-demo',
                            type='primary'
                        ),
                        style={
                            'backgroundColor': 'white'
                        }
                    )
                ],
                style={
                    'height': '600px'
                }
            )
        ],
        style={
            'height': '600px',
            'border': '1px solid rgb(241, 241, 241)'
        }
    ),
    text='回调中'
)

layout = html.Div(children=[
    sidebar_2,
    html.H1(children='Model vs Model', style={'text-align': 'center','font-weight': 'bold'}),
    html.Hr(),
    
    html.H5('How many models you comapre?', style={'margin':10}),
    n_model := dbc.Input(value=3, type='number', style={'margin':10}),
    
    html.Div(id='select_model'),
    
    html.Hr(),
    html.H3(children='Quantitative Indicator'),
    dbc.Row([
      dbc.Col([
          html.H4('Recall'),
          dcc.Graph(figure=fig_recall)
            ]),
      dbc.Col([
          html.H4('MAP'),
          dcc.Graph(figure=fig_map)
    ]),
]),
    dbc.Row([
        dbc.Col([
            html.H4('NDCG'),
          dcc.Graph(figure=fig_ndcg)
          ]),
        dbc.Col([
            html.H4('NDCG'),
          dcc.Graph(figure=fig_popularity)
          ]),
    ]),
],style={
        # 'display': 'flex',
        # 'height': '100%',
        # 'justifyContent': 'center',
        # 'alignItems': 'center'
        })

@callback(
    Output(component_id='select_model', component_property='children'),
    Input(n_model, component_property='value')
)
def update_city_selected(input_value):
    if input_value == None:
        raise PreventUpdate
    model_list = [dbc.DropdownMenu([dbc.DropdownMenuItem(key, n_clicks=0) for key in model_hype_type.keys()],
        # list(model_hype_type.keys()), 
                                   id=f'model_{i}',className='compare_form',style={'margin':10, 'width':'40%'}) for i in range(input_value)]
    model_list.append(dbc.Button('Comapre!', id='Compare_btn',type='submit' ,n_clicks=0, className='compare_form', style={'margin':10, }) )
    result = html.Form(model_list, className= 'compare_form',) 
    # print(result.value)
    return result


@callback(
        Output('map', 'children'),
        Input('compare_btn', 'n_clicks'),
)
def get_quantative_metrics(form):
    params={'model_name': form['model'], 'str_key': form['values']}
    return requests.get(url=f'{API_url}/metric/quantitative/', params=params).json()[0]

@callback(
    Output('sider-custom-trigger-demo', 'collapsed'),
    Input('sider-custom-trigger-button-demo', 'nClicks'),
    State('sider-custom-trigger-demo', 'collapsed'),
    prevent_initial_call=True
)
def sider_custom_trigger_demo(nClicks, collapsed):
    if nClicks:
        return not collapsed

    return dash.no_update