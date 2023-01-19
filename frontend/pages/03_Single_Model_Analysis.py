import streamlit as st
import requests
import pandas
import plotly

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
                                                      'Diversity(jacard)', 'Serendipity(pmi)',
                                                      'Serendipity(jacard)', 'Novelty(rating)',
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
    
sma_plot1, sma_plot2 = st.columns(2)

sma_plot1.markdown('<h4>Serendipity</h4>', unsafe_allow_html=True)

sma_plot1.markdown('<h4>Novelty</h4>', unsafe_allow_html=True)

sma_plot2.markdown('<h4>Diversity</h4>', unsafe_allow_html=True)

sma_plot2.markdown('<h4>Coverage</h4>', unsafe_allow_html=True)

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