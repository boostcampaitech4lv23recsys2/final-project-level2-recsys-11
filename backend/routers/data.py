import pandas as pd

from fastapi import APIRouter

router = APIRouter()


class dataset_info:

    def __init__(self, train_df, item_df, user_df
                    ground_truth, K, item_h_matrix):

        # Dataset
        self.train_df = train_df                                        # interaction data
        self.item_df = item_df                                          # item context data
        self.user_df = user_df
        # self.ground_truth = ground_truth                    
        self.ground_truth = ground_truth.groupby('user').agg(list)      # ground truth for metric evaluation    

        self.train_df.columns = ['user_id', 'item_id', 'rating', 'timestamp', 'origin_timestamp']
        self.item_df.columns = ['item_id', 'movie_title', 'release_year', 'genre']

        # qualitative class 안으로
        # average rating per item
        self.item_mean_df = train_df.groupby('item_id').agg('mean')['rating'] 
        # Diversity - rating matrix
        self.rating_matrix = train_df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)

        # user, item specific information -- profiles: users/items interacted by an item/user
        self.n_user = train_df['user_id'].nunique()
        self.n_item = train_df['item_id'].nunique()
        self.user_profiles = {user: train_df[train_df['user_id'] == user]['item_id'].tolist() for user in train_df['user_id'].unique()} 
        self.item_profiles = {item : train_df[train_df['item_id'] == item]['user_id'].tolist() for item in train_df['item_id'].unique()}

        # genre of an item in dictionary
        self.genre = dict()
        for i,j in zip(item_df['item_id'], item_df['genre']):
            self.genre[i] = j.split(' ')

        # Matrices for item latent vector
        self.item_h_matrix = item_h_matrix
        self.item_item_matrix = self.item_h_matrix @ self.item_h_matrix.T

        # Recommendation list length for each users
        self.K = K

        # Popularity
        self.pop_user_per_item = self.calculate_Popularity_user()
        self.pop_inter_per_item = self.calculate_Popularity_inter()

    def calculate_Popularity_user(self):   # 유저 관점의 popularity
        '''
        return: 각 아이템 번호에 따른 인기도 딕트

        상호작용한 유저 수를 기반으로 인기도 측정
        '''

        pop_user_per_item = (self.train_df['item_id'].value_counts() / self.n_user).to_dict()

        return pop_user_per_item

    def calculate_Popularity_inter(self):    # interaction 관점의 popularity
        '''
        return: 각 아이템 번호에 따른 인기도 딕트

        상호작용 횟수를 기반으로 인기도 측정
        '''

        inter_count_of_items = self.train_df.groupby('item_id').count()['user_id']
        total_len = len(self.train_df)

        pop_inter_per_item = dict()
        for i, j in zip(inter_count_of_items.keys(), inter_count_of_items):
            pop_inter_per_item[i] = j / total_len
        #차라리 total_len을 항상 들고 다니고, 여기는 그냥 상호작용된 횟수만 넣고 다니는 게 좋은가?
        return pop_inter_per_item 

train_df = pd.read_csv('/opt/ml/final-project-level2-recsys-11/dataset/ml-1m/ml-1m.inter', sep='\t')
item_df = pd.read_csv('/opt/ml/final-project-level2-recsys-11/dataset/ml-1m/ml-1m.item', sep='\t')
user_df = pd.read_csv('/opt/ml/final-project-level2-recsys-11/dataset/ml-1m/ml-1m.user', sep='\t')
ground_truth = pd.read_csv('/opt/ml/final-project-level2-recsys-11/dataset/ml-1m/ml-1m.test', sep='\t')


dataset = dataset_info(K=10)