# manage user
# from flask_login import login_user, logout_user, current_user, LoginManager, UserMixin
import os
# manage password hahsing
from werkzeug.security import generate_password_hash, check_password_hash
import dash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
# from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(
    __name__, external_stylesheets=[dbc.themes.JOURNAL, dbc_css], 
    use_pages=True, suppress_callback_exceptions=True
) # 페이지 스타일 변경


username = 'mkdir'
# load_figure_template("journal") # figure 스타일 변경

app.layout = html.Div([
	dash.page_container
])



if __name__ == '__main__':
    app.run_server(debug=True, port=30007, host='0.0.0.0')