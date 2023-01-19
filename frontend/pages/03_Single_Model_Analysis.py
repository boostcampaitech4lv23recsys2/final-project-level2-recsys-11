import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

class ReRanking_Form:
    def __init__(self, key:int) -> None:
        self.key = key
        self.container = st.sidebar.container()
        self.alpha = self.container.radio(label='Select alpha',
                         options=(0, 0.5, 1),
                         key=self.key,
                         horizontal=True
        )
        self.key += 1

        self.obj_func = self.container.selectbox(label='Select object function and distance function',
                                             options=('Diversity(rating)', 'Diversity(item vector)', 
                                                      'Diversity(jacard)', 'Serendipity(rating)',
                                                      'Serendipity(item vector)','Serendipity(jacard)',
                                                      'Novelty(rating)',
                                                      'Novelty(item vector)','Novelty(jacard)'),
                                             key=self.key,
        )
        self.key += 1
        self.container.markdown('---', unsafe_allow_html=True)

    def get_data(self):
        datas = dict()
        datas['alpha'] = self.alpha
        datas['obj_func'] = self.obj_func
        return datas
    

st.markdown(f"<h1 style='text-align: center; color: black;'>Single Model Analysis</h1>", unsafe_allow_html=True)
st.markdown(f"---", unsafe_allow_html=True) 

n_recommends = st.sidebar.number_input('How many recommends you compare?', step = 1, max_value=4, value=2)

before_key = 0
forms = []
for i in range(int(n_recommends)):
    reranking_form = ReRanking_Form(key=before_key)
    before_key += reranking_form.key 
    forms.append(reranking_form.get_data())

st.sidebar.button('Compare!')
exp_div_df = pd.DataFrame(columns = ['model', 'diversity', 'mean', 'colors'])
## TODO : '/Diversity_for_users'로 request 보내서 Diversity plot 그리기 위한 정보 받아오기

## TODO : for문으로 비교할만큼 모델 정보 받아와서 추가하기
exp_div_df.loc[1, :] = ['M1',[0.4, 0.5, 0.5, 0.6, 0.6, 0.7], 0.5, 'red']
exp_div_df.loc[2, :] = ['M2',[0.1, 0.2, 0.2, 0.3, 0.3, 0.4], 0.2, 'blue']
exp_div_df.loc[3, :] = ['M3',[0.6, 0.7, 0.7, 0.8, 0.8, 0.9], 0.8, 'green']

div_plot1, div_plot2 = st.columns(2)

div_plot1.markdown('<h4>Diversity Histogram</h4>', unsafe_allow_html=True)
fig = go.Figure()
for i in exp_div_df.index:    
    fig.add_trace(go.Histogram(
        x=exp_div_df.loc[i,'diversity'],
        marker={'color' : exp_div_df.loc[i,'colors']}
     ))
fig.update_traces(opacity=0.5)
div_plot1.plotly_chart(fig, use_container_width=True)

div_plot2.markdown('<h4>Diversity Bar plot</h4>', unsafe_allow_html=True)
fig = px.bar(exp_div_df, x='model', y='mean', color='model', color_discrete_sequence=exp_div_df['colors'].values)
div_plot2.plotly_chart(fig, use_container_width=True)

## TODO : '/Serendipity_for_users'로 request 보내서 Serendipty plot 그리기 위한 정보 받아오기

ser_plot1, ser_plot2 = st.columns(2)

ser_plot1.markdown('<h4>Serendipity Histogram</h4>', unsafe_allow_html=True)

ser_plot2.markdown('<h4>Serendipity Bar plot</h4>', unsafe_allow_html=True)

## TODO : '/Novelty_for_users'로 request 보내서 Novelty plot 그리기 위한 정보 받아오기

nov_plot1, nov_plot2 = st.columns(2)

nov_plot1.markdown('<h4>Novelty Histogram</h4>', unsafe_allow_html=True)

nov_plot2.markdown('<h4>Novelty Bar plot</h4>', unsafe_allow_html=True)


st.markdown(f"<h2 style='text-align: left; color: black;'>Patterns by side-informations</h2>", unsafe_allow_html=True)

pbs_plot1, pbs_plot2 = st.columns(2)

pbs_plot1.markdown('<h4> User-side</h4>', unsafe_allow_html=True)
user_feat, user_metric, item_feat, item_metric = st.columns(4)
users_feat = ('age', 'job', 'occupation')
user_feat.selectbox("User's feature:", users_feat)

pbs_plot1.line_chart()

metrics = ('Diversity', 'Novelty', 'Serendipity')
user_metric.selectbox("Metric:", metrics)

pbs_plot2.markdown('<h4> Item-side</h4>', unsafe_allow_html=True)
pbs_plot2.line_chart()

items_feat = ('genre', 'director', 'writer')
item_feat.selectbox("Item's feature:", items_feat)

pbs_plot2.line_chart()

metrics = ('Diversity', 'Novelty', 'Serendipity')
item_metric.selectbox("Metric: ", metrics)