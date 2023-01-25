import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px

API_url = 'http://127.0.0.1:8000'

dash.register_page(__name__)

# username = requests.get(f'{API_url}/username').json()

model_hype_type = requests.get(url=f'{API_url}/model_hype_type').json()


exp_df = pd.DataFrame(columns = ['model','recall','ndcg','map','popularity','colors'])

exp_df.loc[1,:] = ['M1',0.1084,0.0847,0.1011,0.0527,'red']
exp_df.loc[2,:] = ['M2',0.1124,0.0777,0.1217,0.0781,'green']
exp_df.loc[3,:] = ['M3',0.1515,0.1022,0.1195,0.0999,'blue']
exp_df.loc[4,:] = ['M4',0.0917,0.0698,0.0987,0.0315,'goldenrod']

fig = px.bar(
            exp_df,
            x = 'model',
            y ='recall',
            color = 'model',
            color_discrete_sequence=exp_df['colors'].values
            )

layout = html.Div(children=[
    html.H1(children='Model vs Model', style={'text-align': 'center','font-weight': 'bold'}),
    html.Hr(),
    
    html.H5('How many models you comapre?'),
    n_model := dcc.Input(value=3, type='number', ),
    
    html.Div(id='select_model'),
    
    html.Hr(),
    html.H3(children='Quantitative Indicator'),
    dbc.Row([
      dbc.Col([
          html.H4('Recall'),
          dcc.Graph(figure=fig)
            ]),
      dbc.Col(
          html.H4('MAP')
          ),
]),
    dbc.Row([
            html.H4('NDCG'),
          dcc.Graph(figure=fig)
    ]),
])

@callback(
    Output(component_id='select_model', component_property='children'),
    Input(n_model, component_property='value')
)
def update_city_selected(input_value):
    return [dcc.Dropdown(list(model_hype_type.keys())) for _ in range(input_value)]