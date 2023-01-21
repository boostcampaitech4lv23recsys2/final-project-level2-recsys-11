import pandas as pd
import streamlit as st
import matplotlib as plt
import viz
import plotly.express as px
import os
import requests
import plotly.io as pio
from design import head, body

API_url = 'http://127.0.0.1:8000'

st.set_page_config(
    page_title="Wᴇʙ4Rᴇᴄ",
    page_icon="😊",
    # layout="wide",
    initial_sidebar_state="expanded",
)

head.set_title('ᴡᴇʙ4ʀᴇᴄ', 'center')

a = st.columns(2)
id = a[0].text_input('Username', )
password = a[0].text_input('Password', type='password')
a[1].text(' ')
a[1].text(' ')

def login():
    params = {'id':id, 'password':password}
    '''
    respone = requests.get(f'{API_url}/login', params=params)
    if respone.status_code == 200:
        show next page
    else:
        a[1].write('wrong username or password')
    
    '''
    
    
    a[1].write('dsf')
    pass
a[1].button('Signin', on_click=login)
def signup():
    import streamlit as st
    
    st.text('sign')
    # head.set_title('Sign up', 'center')

a[0].button('Signup', on_click=signup)