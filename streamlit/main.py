import pandas as pd
import streamlit as st
import matplotlib as plt
import viz
import plotly.express as px
import os
import requests
import plotly.io as pio
from design import head, body


st.set_page_config(
    page_title="Recommendation Model Evaluation",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

df_r = requests.get(
    url = 'http://127.0.0.1:30004/data'
)


df = pd.DataFrame(df_r.json())

head.set_title()

st.markdown('<h3>compare table</h3>', unsafe_allow_html=True) # display_dataframe에서 실행하면 에러 남
body.display_dataframe(df, container_width=True)
st.markdown('---')
body.compare_metric()

plot_r = requests.get(
    url = 'http://127.0.0.1:30004/plot'
)
plotly_fig = pio.from_json(plot_r.json())
st.plotly_chart(plotly_fig)
