import streamlit as st

def set_title():
    title_text = 'Recommendation Model Evaluation'
    st.markdown(f"<h1 style='text-align: center; color: black;'>{title_text}</h1>", unsafe_allow_html=True) # 중앙 정렬 텍스트
    st.markdown(f"---", unsafe_allow_html=True) 