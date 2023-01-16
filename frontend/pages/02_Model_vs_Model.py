import streamlit as st
import altair as alt
import pandas as pd
import requests
import time
import numpy as np
import random

st.set_page_config(layout="wide")

# sidebar settings
class model_form():
    def __init__(self, key):
        self.key = key
        self.container = st.sidebar.container()
        
        hypes = requests.get(url='http://0.0.0.0:8000/model_hype_type').json()
        self.model = self.container.selectbox(label='Select Model',
                         options=hypes.keys(),
                         key=key
        )
        self.key +=1
        self.color = self.container.color_picker('select color', key=self.key)
        self.key += 1
        n = len(hypes[self.model])
        self.n = n
        for plus, hype in enumerate(hypes[self.model], self.key):
            self.container.radio(hype, (1,2,3), key=plus, horizontal=True)
        self.key += self.n
        

forms = []

n_model = st.sidebar.number_input('how many models you compare?', step = 1, max_value=5)
before_param_n = 0
for i in range(int(n_model)):
    # random.
    
    A = model_form(key=before_param_n)
    before_param_n += A.key 
    forms.append(A)
    # forms.append(model_form(key = i))

def plot_models():
    with st.spinner('Wait for drawing..'):
        # requests(url)
        time.sleep(5)
    st.success('done!')
    # if request.get(url):
    #     draw_page()

st.sidebar.button('Compare!', on_click=plot_models)
# st.sidebar.radio('select value', (1,2,3), horizontal=True)



# for form in forms:
#     st.write(form.model)


# header
st.markdown(f"<h1 style='text-align: center; color: black;'>Model vs Model</h1>", unsafe_allow_html=True)
st.markdown(f"---", unsafe_allow_html=True) 

# body: show Quantitative Indicator
st.markdown(f"<h2 style='text-align: left; color: black;'>Quantitative Indicator</h2>", unsafe_allow_html=True)

# plot에 필요한 데이터 프레임 받아오는 함수 작성
exp_df = pd.DataFrame(columns = ['model','recall','ndcg','map','popularity'])

#TODO: data_df = request(url)
# exp_df.loc['모델 고유 번호',:] = ['recall','ndcg','map','popularity']
exp_df.loc[1,:] = ['M1',0.1084,0.0847,0.1011,0.0527]
exp_df.loc[2,:] = ['M2',0.1124,0.0777,0.1217,0.0781]
exp_df.loc[3,:] = ['M3',0.1515,0.1022,0.1195,0.0999]
exp_df.loc[4,:] = ['M4',0.0917,0.0698,0.0987,0.0315]

plot1, plot2 = st.columns(2)

#TODO: 위에서 받은 df를 그래프로 나타내기
plot1.markdown('<h4>Recall</h4>', unsafe_allow_html=True)
bar_chart = alt.Chart(exp_df).mark_bar().encode(
    alt.X('model:O'),
    alt.Y('recall:Q'),
    alt.Color('model:O'),
)
plot1.altair_chart(bar_chart, use_container_width=True)

plot1.markdown('<h4>NDCG</h4>', unsafe_allow_html=True)
bar_chart = alt.Chart(exp_df).mark_bar().encode(
    alt.X('model:O'),
    alt.Y('ndcg:Q'),
    alt.Color('model:O'),
)
plot1.altair_chart(bar_chart, use_container_width=True)

plot2.markdown('<h4>MAP</h4>', unsafe_allow_html=True)
bar_chart = alt.Chart(exp_df).mark_bar().encode(
    alt.X('model:O'),
    alt.Y('map:Q'),
    alt.Color('model:O'),
)
plot2.altair_chart(bar_chart, use_container_width=True)

plot2.markdown('<h4>Popularity</h4>', unsafe_allow_html=True)
bar_chart = alt.Chart(exp_df).mark_bar().encode(
    alt.X('model:O'),
    alt.Y('popularity:Q'),
    alt.Color('model:O'),
)
plot2.altair_chart(bar_chart, use_container_width=True)

st.markdown(f"<h2 style='text-align: left; color: black;'>Quantitative  Indicator</h2>", unsafe_allow_html=True)

plot3, plot4 = st.columns(2)

plot3.markdown('<h4>Diversity</h4>', unsafe_allow_html=True)
# plot3.line_chart() #TODO: 위에서 받은 df를 그래프로 나타내기
plot3.markdown('<h4>Novelty</h4>', unsafe_allow_html=True)
# plot3.line_chart()

plot4.markdown('<h4>Serendipity</h4>', unsafe_allow_html=True)
# plot4.line_chart()
plot4.markdown('<h4>Coverage</h4>', unsafe_allow_html=True)
# plot4.line_chart()

def plot_Recall():
    pass
