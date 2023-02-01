from dash import dcc


user_state = dcc.Store(id='user_state', storage_type='session')

def get_session():
    return user_state