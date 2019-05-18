import numpy as np
import pandas as pd

# TODO: build cascade file

# building the user_data file
repost = pd.read_csv('../data/repost.csv',
                     header=None,
                     names=['original_status_id', 'status_id', 'user_id', 'topic', 'original_user_id'])
# due to the lack of time in repost, the repost rate is unknown.
repost_count = repost.groupby(by=['user_id', 'original_user_id']).size()
rc = pd.DataFrame(repost_count)
rc.reset_index(inplace=True)
rc.columns = ['user_id', 'original_user_id', 'repostn']
# rc is DataFrame
# user_id | original_user_id | repostn

status_new = pd.read_csv('../data/status_new.csv', header=None,
                         names=['status_id', 'retweet_count', 'favorite_count', 'created_at', 'text', 'catch_time',
                                'user_id'])
status = pd.read_csv('../data/status.csv', header=None,
                     names=['status_id', 'retweet_count', 'favorite_count', 'created_at', 'text', 'catch_time',
                            'user_id'])
statusall = status.append(status_new, ignore_index=True)
status_count = statusall.groupby(by=['user_id']).size()
sc = pd.DataFrame(status_count)
sc.reset_index(inplace=True)
sc.columns = ['user_id', 'statusn']
# sc is DataFrame
# user_id | statusn

# If the sampling is uniform, the number of statuses we collect from a user is only a constant away from the post_rate
friends = pd.read_csv('../data/friends.csv', header=None, names=['user_id', 'friend_id'])
inflow = pd.merge(friends, sc, left_on='friend_id', right_on='user_id', how='left')
inflow.drop('user_id_y', axis=1, inplace=True)
inflow.columns = ['user_id', 'friend_id', 'statusn']
inflow2 = pd.merge(inflow, rc, left_on=['user_id', 'friend_id'], right_on=['user_id', 'original_user_id'], how='left')
inflow2.drop('original_user_id', axis=1, inplace=True)
inflow2.loc[inflow2.statusn.isnull(), 'statusn'] = 0
inflow2.loc[inflow2.repostn.isnull(), 'repostn'] = 0
inflow2['scxrepost'] = inflow2['statusn'] * inflow2['repostn']

friend_avg_inflow = inflow2.groupby('user_id').agg({
    'statusn': np.sum,
    'repostn': np.sum,
    'scxrepost': np.sum,
})

friend_avg_inflow.reset_index(inplace=True)
friend_avg_inflow.columns = ['user_id', 'inflow', 'repost_sum', 'status_weighted_sum']

friend_avg_inflow['avg_inflow'] = friend_avg_inflow['repost_sum'] / friend_avg_inflow['status_weighted_sum']
friend_avg_inflow.loc[friend_avg_inflow.status_weighted_sum == 0, 'avg_inflow'] = 0

# friend_avg_inflow is DataFrame
# user_id | inflow | repost_sum | status_weighted_sum | avg_inflow

temp = pd.merge(friend_avg_inflow, sc, on='user_id', how='left')
temp.loc[temp.statusn.isnull(), 'statusn'] = 0
temp.drop(['repost_sum', 'status_weighted_sum'], axis=1, inplace=True)
temp.columns = ['user_id', 'inflow', 'avg_inflow', 'outflow']

user1 = pd.read_csv('../data/users.csv', header=None,
                    names=['user_id', 'friend_count', 'follower_count', 'listed_count',
                           'status_count', 'favorites_count', 'created_at', 'name', 'verified'],
                    usecols=[0, 1, 2])
user2 = pd.read_csv('../data/users_new.csv', header=None,
                    names=['user_id', 'friend_count', 'follower_count', 'listed_count',
                           'status_count', 'favorites_count', 'created_at', 'name', 'verified'],
                    usecols=[0, 1, 2])
users = user1.append(user2, ignore_index=True)

user_data = pd.merge(users, temp, on='user_id', how='left')
user_data.loc[user_data.inflow.isnull(), 'inflow'] = 0
user_data.loc[user_data.avg_inflow.isnull(), 'avg_inflow'] = 0
user_data.loc[user_data.outflow.isnull(), 'outflow'] = 0

user_data.to_csv('../data/micro-macro-user-features.csv')
