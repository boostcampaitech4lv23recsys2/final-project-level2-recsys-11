import pandas as pd

class dataset_info:

    def __init__(self, train_df, item_df, user_df, ground_truth, K):
        
        # Dataset
        self.train_df = train_df                                        # interaction data
        self.item_df = item_df                                          # item context data
        self.user_df = user_df
        # self.ground_truth = ground_truth                    
        self.ground_truth = ground_truth.groupby('user').agg(list)      # ground truth for metric evaluation    

        self.train_df.columns = ['user_id', 'item_id', 'rating', 'timestamp'] #, 'origin_timestamp']
        self.item_df.columns = ['item_id', 'movie_title', 'release_year', 'genre']

        # Recommendation list length for each users
        self.K = K
        
        # user, item specific information -- profiles: users/items interacted by an item/user
        self.n_user = train_df['user_id'].nunique()
        self.n_item = train_df['item_id'].nunique()
        self.user_profiles = {user: train_df[train_df['user_id'] == user]['item_id'].tolist() for user in train_df['user_id'].unique()} 
        self.item_profiles = {item : train_df[train_df['item_id'] == item]['user_id'].tolist() for item in train_df['item_id'].unique()}

        # Popularity
        self.pop_user_per_item = self.calculate_Popularity_user()
        self.pop_inter_per_item = self.calculate_Popularity_inter()

        # qualitative class 안으로
        # average rating per item
        self.item_mean_df = train_df.groupby('item_id').agg('mean')['rating'] 
        # Diversity - rating matrix
        self.rating_matrix = train_df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)

        # genre of an item in dictionary
        self.genre = dict()
        for i,j in zip(item_df['item_id'], item_df['genre']):
            self.genre[i] = j.split(' ')

    def calculate_Popularity_user(self):   # 유저 관점의 popularity - default
        '''
        popularity = (item과 상호작용한 유저 수) / (전체 유저 수) 

        return: dict('item_id': popularity, 'item_id2': popularity2, ...)
        '''

        pop_user_per_item = (self.train_df['item_id'].value_counts() / self.n_user).to_dict()

        return pop_user_per_item

    def calculate_Popularity_inter(self):    # interaction 관점의 popularity
        '''
        popularity = (item과 상호작용한 유저 수 = 상호작용 횟수) / (전체 상호작용 수) 

        return: dict('item_id': popularity, 'item_id2': popularity2, ...)
        '''

        inter_count_of_items = self.train_df.groupby('item_id').count()['user_id']
        total_len = len(self.train_df)

        pop_inter_per_item = dict()
        for i, j in zip(inter_count_of_items.keys(), inter_count_of_items):
            pop_inter_per_item[i] = j / total_len
        
        return pop_inter_per_item 


train_df = pd.read_csv('/opt/ml/final-project-level2-recsys-11/dataset/ml-1m/ml-1m.inter', sep='\t')
item_df = pd.read_csv('/opt/ml/final-project-level2-recsys-11/dataset/ml-1m/ml-1m.item', sep='\t')
user_df = pd.read_csv('/opt/ml/final-project-level2-recsys-11/dataset/ml-1m/ml-1m.user', sep='\t')
ground_truth = pd.read_csv('/opt/ml/final-project-level2-recsys-11/dataset/ml-1m/ml-1m.test', sep='\t')


dataset = dataset_info(train_df=train_df, item_df=item_df, user_df=user_df, 
                        ground_truth=ground_truth, K=10)