import os

import pandas as pd

# load origin files
dir_path = os.path.dirname(os.path.abspath(__file__))

ratings = pd.read_csv(os.path.join(dir_path, 'ratings.dat'), sep='::', engine='python', encoding='latin-1', header=None)
users = pd.read_csv(os.path.join(dir_path, 'users.dat'), sep='::', engine='python', encoding='latin-1', header=None)
items = pd.read_csv(os.path.join(dir_path, 'movies.dat'), sep='::', engine='python', encoding='latin-1', header=None)

ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']
users.columns = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
items.columns = ['item_id', 'title', 'genres']

# ratings preprocess
ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s').dt.strftime('%Y%m')

# user preprocess
users['zip_code_2'] = users['zip_code'].apply(lambda s: s[:2])

# item preprocess
items['year'] = items['title'].apply(lambda s: s.split()[-1].replace('(', '').replace(')', ''))
items['title'] = items['title'].apply(lambda s: s.split('(')[0].strip())
items['genres'].apply(lambda s: s.replace('|', ' '))

# detach ground truth
ground_truth_indice = ratings.sort_values(by=['user_id', 'timestamp']).groupby('user_id')['item_id'].tail(1).index
train_ratings = ratings.drop(index=ground_truth_indice).reset_index(drop=True)
test_ratings = ratings.loc[ground_truth_indice].reset_index(drop=True)


# save all
train_ratings.to_csv(os.path.join(dir_path, 'train_ratings.csv'), sep='\t', index=False)
test_ratings.to_csv(os.path.join(dir_path, 'test_ratings.csv'), sep='\t', index=False)
users.to_csv(os.path.join(dir_path, 'users.csv'), sep='\t', index=False)
items.to_csv(os.path.join(dir_path, 'items.csv'), sep='\t', index=False)

