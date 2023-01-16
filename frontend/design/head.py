import streamlit as st

def set_title(text:str, text_align:str='justify'):
    title_text = 'Recommendation Model Evaluation'
    st.markdown(f"<h1 style='text-align: {text_align}; color: black;'>{text}</h1>", unsafe_allow_html=True) # 중앙 정렬 텍스트
    st.markdown(f"---", unsafe_allow_html=True) 
    