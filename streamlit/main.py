import pandas as pd
import streamlit as st
import matplotlib as plt
import viz
import plotly.express as px
import os

df = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')
output_path = '/opt/ml/final_project/data/rec_output'

st.set_page_config(
    page_title="Recommendation Model Evaluation",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)


multi_plot = st.container()

mcol1, mcol2, mcol3 = multi_plot.columns(3)
'''
임베딩 벡터 시각화 아이디어
'''
# mcol1.markdown(f"<h1 style='text-align: center; color: black;'>{option}</h1>", unsafe_allow_html=True) # 중앙 정렬 텍스트
mcol1.selectbox(label='select value',options=df.columns, key=1)
# mcol1.header(f'{option}+1', align='center)
mcol1.pyplot(fig = viz.histogram(df.user))