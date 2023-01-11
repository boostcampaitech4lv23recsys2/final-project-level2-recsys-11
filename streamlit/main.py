import pandas as pd
import streamlit as st
import matplotlib as plt
import viz
import plotly.express as px
import os
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

@st.cache
def load_data(path):
    df = pd.read_csv(path)[:100]
    return df

df = load_data('/opt/ml/input/data/train/train_ratings.csv')[:100]

head.set_title()

st.markdown('<h3>compare table</h3>', unsafe_allow_html=True)
body.display_dataframe(df, container_width=True)
st.markdown('---')
body.compare_metric()
multi_plot = st.container()

mcol1, mcol2, mcol3 = multi_plot.columns(3)
'''
임베딩 벡터 시각화 아이디어
'''
if 'column' not in st.session_state:
    st.session_state.column = 'user'
# mcol1.markdown(f"<h1 style='text-align: center; color: black;'>{option}</h1>", unsafe_allow_html=True) # 중앙 정렬 텍스트
st.session_state.column = mcol1.selectbox(label='select value',options=df.columns, key=1)
# mcol1.header(f'{option}+1', align='center)
mcol1.pyplot(fig = viz.histogram(df[st.session_state.column]))