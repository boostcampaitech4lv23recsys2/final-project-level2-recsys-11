import streamlit as st
import pandas as pd
import numpy as np

def display_dataframe(df:pd.DataFrame, container_width:bool=False) -> None:
    st.dataframe(df, use_container_width=container_width)
    
def compare_metric():
    pass