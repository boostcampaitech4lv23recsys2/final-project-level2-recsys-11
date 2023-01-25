# manage user
from flask_login import login_user, logout_user, current_user, LoginManager, UserMixin
import os
# manage password hahsing
from werkzeug.security import generate_password_hash, check_password_hash
import dash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate

app = Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], 
    use_pages=True, suppress_callback_exceptions=True
) # 페이지 스타일 변경


username = 'mkdir'
load_figure_template("bootstrap") # figure 스타일 변경


app.layout = html.Div([
	html.H1('Web4Rec'),
    html.Hr(),
    # html.Div(
    #     [
    #         html.Div(
    #             dcc.Link(
    #                 f"{page['name']}", href=page["relative_path"]
    #             ),
    #             style={'font-size':30}
    #         )
    #         for page in dash.page_registry.values()
    #     ]
    # ),
    
    
	dash.page_container
])



if __name__ == '__main__':
    app.run_server(debug=True, port=30007, host='0.0.0.0')