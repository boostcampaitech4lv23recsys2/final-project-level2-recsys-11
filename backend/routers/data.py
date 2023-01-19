import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class dataset_info:

    def __init__(self, train_df, item_df, user_df, ground_truth, K):
        '''
        train_df : 학습 데이터 (데이터 프레임)
        item_df : 아이템 사이드 정보 데이터(데이터 프레임)
        user_df : 유저 사이드 정보 데이터(데이터 프레임)
        ground_truth : 정답 데이터 (데이터 프레임)
        item_h_matrix :
        K : 유저별 추천 개수.int

        '''

        # Dataset
        self.train_df = train_df                                        # interaction data
        self.item_df = item_df                                          # item context data
        self.user_df = user_df

        ground_truth.columns = ['user_id', 'item_id']
        self.ground_truth = ground_truth.groupby('user_id').agg(list)               # ground truth for metric evaluation
        self.ground_truth = self.ground_truth.applymap(lambda x: np.array(x))       # pd.DataFrame

        self.train_df.columns = ['user_id', 'item_id', 'rating', 'timestamp','origin_timestamp']
        self.item_df.columns = ['item_id', 'movie_title', 'release_year', 'genre']

        # Recommendation list length for each users
        self.K = K

        # user, item specific information -- profiles: users/items interacted by an item/user
        self.n_user = train_df['user_id'].nunique()
        self.n_item = train_df['item_id'].nunique()
        tail_df = train_df.groupby('user_id').tail(20)

        self.user_profiles = {user: tail_df[tail_df['user_id'] == user]['item_id'].tolist() for user in tail_df['user_id'].unique()}
        # self.item_profiles = {item : train_df[train_df['item_id'] == item]['user_id'].tolist() for item in train_df['item_id'].unique()}

        # Popularity
        self.pop_user_per_item = self.calculate_Popularity_user()
        self.pop_inter_per_item = self.calculate_Popularity_inter()

        # qualitative class 안으로
        # average rating per item
        self.item_mean_df = train_df.groupby('item_id').agg('mean')['rating']
        # Diversity - rating matrix
        self.rating_matrix = train_df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)

        # Serendiptiy - PMI
        self.implicit_matrix = self.rating_matrix.copy()
        # self.implicit_matrix[self.implicit_matrix > 0] = 1
        # self.pmi_df = self.implicit_matrix.T @ self.implicit_matrix
        # self.item_lst = self.pmi_df.columns
        
        train_mat = csr_matrix((np.ones(len(self.train_df)), (self.train_df['user_id'], self.train_df['item_id'])))
        self.pmi_df = train_mat.T @ train_mat
        tmp1 = self.pmi_df.toarray()
        

        #PMI지표 행렬 만들기. 모든 아이템 쌍에 대해 PMI값을 구합니다.
        # tmp1 = self.pmi_df.to_numpy() # 행렬 연산 편의를 위해 넘파이로 만듭니다. 위에 item_lst라는 변수를 통해 이후 다시 DataFrame으로 만들어줍니다.
        tmp1 = tmp1 + 1e-6 # log에 0이 들어가는 것을 막기 위해 작은 값을 더합니다.
        tmp2 = np.triu(tmp1) * self.n_user + np.triu(np.ones_like(tmp1), k=1).T # 하삼각 행렬은 1로 만들고 나머지에 대해 전체 유저를 곱함
        tmp2 = tmp2 / (tmp2.diagonal() / self.n_user) # 행렬 내 모든 값에 대해 대각 행렬 값을 나눈다. 대각행렬 부분은 한 아이템에 대해 상호작용한 유저수를 나타낸다.
        tmp2 = tmp2 * tmp2.T # 전치 행렬과의 아다마르 곱을 통해 (P_ij * n_user) / (P_i * P_j)를 만들어준다.
        tmp2 = np.log2(tmp2) / -np.log2(tmp1 / self.n_user) # PMI 최종 계산 -1~1 사이의 값
        tmp2 = (1 - tmp2) / 2 # 0과 1사이의 값으로 바꿔주는 동시에 음수를 취해준다.
        np.fill_diagonal(tmp2, val=0)
        # self.pmi_matrix = pd.DataFrame(tmp2, index=self.item_lst, columns=self.item_lst)
        self.pmi_matrix = pd.DataFrame(tmp2)
        self.pmi_matrix.columns.name = 'item_id'
        self.pmi_matrix.index.name = 'item_id'
        #i행 j열에 있는 원소는 아이템 i와 j의 정규화된 PMI 값입니다.

        # genre of an item in dictionary
        self.genre = dict()
        for i,j in zip(item_df['item_id'], item_df['genre']):
            self.genre[i] = j.split(' ')

        uniq_genre = set()
        genre = dict()
        for i,j in zip(item_df['item_id'], item_df['genre']):
            genre[i] = j.split(' ')
            uniq_genre |= set(genre[i])
        genre2id = {j:i for i,j in enumerate(uniq_genre)}
        genre_df = item_df.set_index('item_id').drop(['movie_title', 'release_year', 'genre'], axis=1)
        genre_df = genre_df.reindex(columns=uniq_genre)
        for i in genre:
            lst = np.array([0 for _ in range(18)])
            lst[[genre2id[j] for j in genre[i]]] = 1
            genre_df.loc[i] = lst
        genre_matrix = genre_df @ genre_df.T
        genre_item_lst = genre_matrix.columns
        tmp1 = genre_matrix.to_numpy()
        tmp2 = np.triu(tmp1)
        tmp2 = tmp2 - tmp2.diagonal()
        tmp2 = -(tmp2 + tmp2.T)
        tmp3 = 1 - (tmp1 / (tmp2 + 1e-6))
        np.fill_diagonal(tmp3, val=0)
        self.jaccard_matrix = pd.DataFrame(tmp3, index=genre_item_lst, columns=genre_item_lst)

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


dataset = dataset_info(train_df=train_df, item_df=item_df, user_df=user_df, ground_truth=ground_truth, K=30) # collector에서 k지정 필요
