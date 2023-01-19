import streamlit as st
import altair as alt
import pandas as pd
import plotly.express as px
import requests
import time
import numpy as np
import random
import copy

st.set_page_config(layout="wide")

API_url = 'http://127.0.0.1:8000'
# sidebar settings
class Model_Form:
    def __init__(self, key:int) -> None:
        self.key = key
        self.container = st.sidebar.container()
        self.container.markdown('---', unsafe_allow_html=True)
        
        @st.cache
        def get_hype_type():
            return requests.get(url=f'{API_url}/model_hype_type').json()
        model_hype_type = get_hype_type()
        
        self.model = self.container.selectbox(label='Select Model',
                         options=model_hype_type.keys(),
                         key=self.key
        )
        self.key += 1
        self.color = self.container.color_picker('select color', key=self.key)
        self.key += 1
        n = len(model_hype_type[self.model])
        self.n = n
        
        # self.selected_values = dict()
        self.selected_values = []
        for plus, hype in enumerate(model_hype_type[self.model], self.key):
            # self.selected_values[hype] = self.container.radio(hype, model_hype_type[self.model][hype], key=plus, horizontal=True)
            self.selected_values.append(self.container.radio(hype, model_hype_type[self.model][hype], key=plus, horizontal=True))
        self.key += self.n
        # self.container.markdown('---', unsafe_allow_html=True)

    def get_data(self):
        datas = dict()
        datas['model'] = self.model
        datas['color'] = self.color
        datas['values'] = '_'.join(self.selected_values)
        return datas


#header
st.markdown(f"<h1 style='text-align: center; color: black;'>Model vs Model</h1>", unsafe_allow_html=True)
st.markdown(f"---", unsafe_allow_html=True) 

n_model = st.sidebar.number_input('How many models you compare?', step = 1, max_value=5, value=2)

before_param_n = 0
model_forms = []
for i in range(int(n_model)):
    model_form = Model_Form(key=before_param_n)
    before_param_n += model_form.key 
    model_forms.append(model_form.get_data())

def plot_models():
    st.markdown(f"<h1 style='text-align: center; color: black;'>Model vs Model</h1>", unsafe_allow_html=True)
    st.markdown(f"---", unsafe_allow_html=True) 
    exp_df = pd.DataFrame(columns = ['model','recall','map','ndcg','avg_popularity','coverage'])

    @st.cache(show_spinner=False)
    def get_quantative_metrics(form):
        params={'model_name': form['model'], 'str_key': form['values']}
        return requests.get(url=f'{API_url}/metric/quantitative/', params=params).json()[0]
    
    for i, form in enumerate(model_forms):
        with st.spinner('Wait for drawing..'):
            metrics = get_quantative_metrics(form)
        # st.write(metrics)
        exp_df = exp_df.append({'model': f'M{i}','recall':metrics['recall'], 'ndcg':metrics['ndcg'],
                                'map':metrics['map'], 'popularity':metrics['avg_popularity'], 'coverage':metrics['coverage'], 'colors': form['color']},
                                ignore_index=True)

    # body: show Quantitative Indicator
    st.markdown(f"<h2 style='text-align: left; color: black;'>Quantitative Indicator</h2>", unsafe_allow_html=True)

    # plot_col = st.columns(2)
    plot1, plot2 = st.columns(2)

    # def draw_plot(plot_name:str, metric:str, col_num:st.container) -> None:
        
    #     pass
    
    plot1.markdown('<h4>Recall</h4>', unsafe_allow_html=True)
    fig = px.bar(
        exp_df,
        x = 'model',
        y = 'recall',
        color = 'model',
        color_discrete_sequence=exp_df['colors'].values
        )
    plot1.plotly_chart(fig, use_container_width=True)

    plot1.markdown('<h4>NDCG</h4>', unsafe_allow_html=True)
    fig = px.bar(
        exp_df,
        x = 'model',
        y = 'ndcg',
        color = 'model',
        color_discrete_sequence=exp_df['colors'].values
        )
    plot1.plotly_chart(fig, use_container_width=True)

    plot2.markdown('<h4>MAP</h4>', unsafe_allow_html=True)
    fig = px.bar(
        exp_df,
        x = 'model',
        y = 'map',
        color = 'model',
        color_discrete_sequence=exp_df['colors'].values
        )
    plot2.plotly_chart(fig, use_container_width=True)
    st.success('done!')
    
st.sidebar.button('Compare!', on_click=plot_models)

# header


# # body: show Quantitative Indicator
# st.markdown(f"<h2 style='text-align: left; color: black;'>Quantitative Indicator</h2>", unsafe_allow_html=True)

# # plot에 필요한 데이터 프레임 받아오는 함수 작성
# exp_df = pd.DataFrame(columns = ['model','recall','ndcg','map','popularity','colors'])

# #TODO: data_df = request(url)
# # exp_df.loc['모델 고유 번호',:] = ['recall','ndcg','map','popularity']
# exp_df.loc[1,:] = ['M1',0.1084,0.0847,0.1011,0.0527,'red']
# exp_df.loc[2,:] = ['M2',0.1124,0.0777,0.1217,0.0781,'green']
# exp_df.loc[3,:] = ['M3',0.1515,0.1022,0.1195,0.0999,'blue']
# exp_df.loc[4,:] = ['M4',0.0917,0.0698,0.0987,0.0315,'goldenrod']

# plot1, plot2 = st.columns(2)

# #TODO: 위에서 받은 df를 그래프로 나타내기
# plot1.markdown('<h4>Recall</h4>', unsafe_allow_html=True)
# fig = px.bar(
#     exp_df,
#     x = 'model',
#     y = 'recall',
#     color = 'model',
#     color_discrete_sequence=exp_df['colors'].values
#     )
# plot1.plotly_chart(fig, use_container_width=True)


# plot1.markdown('<h4>NDCG</h4>', unsafe_allow_html=True)
# fig = px.bar(
#     exp_df,
#     x = 'model',
#     y = 'ndcg',
#     color = 'model',
#     color_discrete_sequence=exp_df['colors'].values
#     )
# plot1.plotly_chart(fig, use_container_width=True)


# plot2.markdown('<h4>MAP</h4>', unsafe_allow_html=True)
# fig = px.bar(
#     exp_df,
#     x = 'model',
#     y = 'map',
#     color = 'model',
#     color_discrete_sequence=exp_df['colors'].values
#     )
# plot2.plotly_chart(fig, use_container_width=True)

# plot2.markdown('<h4>Popularity</h4>', unsafe_allow_html=True)
# fig = px.bar(
#     exp_df,
#     x = 'model',
#     y = 'popularity',
#     color = 'model',
#     color_discrete_sequence=exp_df['colors'].values
#     )
# plot2.plotly_chart(fig, use_container_width=True)


# st.markdown(f"<h2 style='text-align: left; color: black;'>Quantitative  Indicator</h2>", unsafe_allow_html=True)

# plot3, plot4 = st.columns(2)

# plot3.markdown('<h4>Diversity(jaccard)</h4>', unsafe_allow_html=True)
# # plot3.line_chart() #TODO: 위에서 받은 df를 그래프로 나타내기
# plot3.markdown('<h4>Novelty(jaccard)</h4>', unsafe_allow_html=True)
# # plot3.line_chart()

# plot4.markdown('<h4>Serendipity(jaccard)</h4>', unsafe_allow_html=True)
# # plot4.line_chart()
# plot4.markdown('<h4>Coverage</h4>', unsafe_allow_html=True)
# # plot4.line_chart()
