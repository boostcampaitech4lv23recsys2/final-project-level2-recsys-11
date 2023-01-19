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
    '''
    원하는 실험을 선택할 수 있도록 모델 및 하이퍼 파라미터 값을 나타내주는 form

    key: input(ex. radio, selectbox) 개수가 가변적으로 변할 때 
         streamlit에서는 input 별로 key를 정해줘야 함.
         중요하진 않음
    '''
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
        r = lambda: random.randint(0,255)
        random_color = '#%02X%02X%02X' % (r(),r(),r())
        self.color = self.container.color_picker('select color', key=self.key, value=random_color)
        
        self.key += 1
        n = len(model_hype_type[self.model])
        self.n = n
        
        self.selected_values = []
        for plus, hype in enumerate(model_hype_type[self.model], self.key):
            self.selected_values.append(self.container.radio(hype, model_hype_type[self.model][hype], key=plus, horizontal=True))
        self.key += self.n

    def get_data(self):
        '''
        form에 유저가 입력한 값을 dict 형태로 뱉어줌
        values는 str_key의 형태로 들어가있음
        '''
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
    '''
    Compare! 버튼을 눌렀을 때 실행되는 함수. 페이지 전체가 다시 그려지기 때문에 주의해야 함
    '''
    st.markdown(f"<h1 style='text-align: center; color: black;'>Model vs Model</h1>", unsafe_allow_html=True)
    st.markdown(f"---", unsafe_allow_html=True) 
    exp_df = pd.DataFrame(columns = ['model','recall','map','ndcg','avg_popularity','coverage'])

    @st.cache(show_spinner=False)
    def get_quantative_metrics(form):
        params={'model_name': form['model'], 'str_key': form['values']}
        return requests.get(url=f'{API_url}/metric/quantitative/', params=params).json()[0]
    
    for i, form in enumerate(model_forms):
        with st.spinner('Please wait for drawing..'):
            metrics = get_quantative_metrics(form)
        exp_df = exp_df.append({'model': f'M{i}','recall':metrics['recall'], 'ndcg':metrics['ndcg'],
                                'map':metrics['map'], 'popularity':metrics['avg_popularity'], 'coverage':metrics['coverage'], 'colors': form['color']},
                                ignore_index=True)

    # body: show Quantitative Indicator
    st.markdown(f"<h2 style='text-align: left; color: black;'>Quantitative Indicator</h2>", unsafe_allow_html=True)

    plot_col = st.columns(2)

    def draw_plot(plot_name:str, metric:str, col_num:int) -> None:
        '''
        다양한 metric을 그리는 함수

        plot_name(str): plot 위에 작성되는 마크다운 문자열
        metric(str): 그래프로 나타낼 metric. exp_df의 colum 이름에 맞게 넣어야 함
        col_num(int): streamlit에서 나눈 컬럼 번호
        '''
        plot_col[col_num].markdown(f'<h4>{plot_name}</h4>', unsafe_allow_html=True)
        fig = px.bar(
            exp_df,
            x = 'model',
            y = metric,
            color = 'model',
            color_discrete_sequence=exp_df['colors'].values
            )
        plot_col[col_num].plotly_chart(fig, use_container_width=True)
    
    plot_list = [
        ('Recall', 'recall', 0),
        ('NDCG', 'ndcg', 0),
        ('MAP', 'map', 1),
        ('Popularity', 'popularity', 1),
        ]
    for plot_name, metric, col_num in plot_list:
        draw_plot(plot_name, metric, col_num)
    
st.sidebar.button('Compare!', on_click=plot_models)


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
