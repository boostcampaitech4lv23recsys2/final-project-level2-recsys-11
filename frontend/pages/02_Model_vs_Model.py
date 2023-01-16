import streamlit as st
import requests
import time

st.set_page_config(layout="wide")

# sidebar settings
class model_form():
    def __init__(self, key, n_hype):
        # radom.color()
        self.model = st.sidebar.selectbox(label='Select Model',
                         options=['BPR', 'EASE'],
                         key=key
        )
        self.hype = dict()
        
    

n_model = st.sidebar.number_input('how many models you compare?', step = 1, max_value=5)
for i in range(int(n_model)):
    model_form(i, 3)

def plot_models():
    with st.spinner('Wait for drawing..'):
        time.sleep(5)
    st.success('done!')
    # if request.get(url):
    #     draw_page()

st.sidebar.button('Compare!', on_click=plot_models)
st.sidebar.radio('select value', (1,2,3), horizontal=True)

# header
st.markdown(f"<h1 style='text-align: center; color: black;'>Model vs Model</h1>", unsafe_allow_html=True)
st.markdown(f"---", unsafe_allow_html=True) 

# body: show Quantitative Indicator
st.markdown(f"<h2 style='text-align: left; color: black;'>Quantitative Indicator</h2>", unsafe_allow_html=True)

# plot에 필요한 데이터 프레임 받아오는 함수 작성
#TODO: data_df = request(url)


plot1, plot2 = st.columns(2)

plot1.markdown('<h4>Recall</h4>', unsafe_allow_html=True)
plot1.line_chart() #TODO: 위에서 받은 df를 그래프로 나타내기
plot1.markdown('<h4>NDCG</h4>', unsafe_allow_html=True)
plot1.line_chart()

plot2.markdown('<h4>MAP</h4>', unsafe_allow_html=True)
plot2.line_chart()
plot2.markdown('<h4>Popularity</h4>', unsafe_allow_html=True)
plot2.line_chart()

st.markdown(f"<h2 style='text-align: left; color: black;'>Quantitative  Indicator</h2>", unsafe_allow_html=True)

plot3, plot4 = st.columns(2)

plot3.markdown('<h4>Diversity</h4>', unsafe_allow_html=True)
plot3.line_chart() #TODO: 위에서 받은 df를 그래프로 나타내기
plot3.markdown('<h4>Novelty</h4>', unsafe_allow_html=True)
plot3.line_chart()

plot4.markdown('<h4>Serendipity</h4>', unsafe_allow_html=True)
plot4.line_chart()
plot4.markdown('<h4>Coverage</h4>', unsafe_allow_html=True)
plot4.line_chart()

def plot_Recall():
    pass
