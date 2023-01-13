from recbole.quick_start.quick_start import load_data_and_model
import pandas as pd
import streamlit as st


class exp_data():
    def __init__(self, pth_path, data_path):
        '''
        n_users, n_items
        user_profiles: user의 rating 기록
        item_profiles: item의 rating 기록
        '''
        # self.pth_path = pth_path
        # self.data_path = data_path
        self.load_data(pth_path, data_path)
    
    @st.cache
    def load_data(self, pth_path, data_path):
        config, _, dataset, train_data, valid_data, test_data = load_data_and_model(pth_path, data_path)
        self.context_data = pd.read_csv(data_path, sep='\t')
        
        
if __name__ == '__main__':
    pth_path = '/opt/ml/final-project-level2-recsys-11/RecBole/saved/EASE-Jan-11-2023_10-27-42.pth'
    data_path = '/opt/ml/final-project-level2-recsys-11/dataset/ml-1m'
    exp_data(pth_path, data_path)