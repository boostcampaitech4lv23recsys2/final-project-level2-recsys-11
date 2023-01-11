import streamlit as st
import pandas as pd
import numpy as np

def display_dataframe(df:pd.DataFrame, container_width:bool=False) -> None:
    st.dataframe(df, use_container_width=container_width)
    
def compare_metric():
    model1, model2 = st.columns(2)
    
    # model1_name = 'BPR_JAN_01_123213'
    model1_name = model1.selectbox(' ', ['BPR_JAN_01_123213', '@#(RFJDSI'])
    
    # model2_name = 'BPR_JAN_01_512413'
    model2_name = model2.selectbox(' ', ['BPR_JAN_01_512413', '@#(RFJDSI'])
    
    METRIC_NUM = 2
    a = st.columns(METRIC_NUM *2)
    # m1_metric1, m1_metric2 = model1.columns(METRIC_NUM)
    # m2_metric1, m2_metric2 = model2.columns(METRIC_NUM)
    # print(a)
    # st.write(a)
    m1_recall = 123
    m2_recall = 234
    m1_precision = 0.87
    m2_precision = 0.80
    
    a[0].metric(label="Recall", value=f"{m1_recall}", delta=f"{(m1_recall - m2_recall):.2f}")
    a[1].metric(label="Precision", value=f"{m2_precision}", delta=f"{(m2_precision - m1_precision):.2f}")
    a[2].metric(label="Recall", value=f"{m1_recall}", delta=f"{(m1_recall - m2_recall):.2f}")
    a[3].metric(label="Precision", value=f"{m2_precision}", delta=f"{(m2_precision - m1_precision):.2f}")
    