import pandas as pd
import numpy as np
from datetime import timedelta
from pickle import load, dump


def create_basic():
    f1 = open('../data/user_profile1.txt')
    f2 = open('../data/user_profile2.txt')

    lines = f1.readlines()
    lines2 = f2.readlines()
    lines.extend(lines2)
    f1.close()
    f2.close()
    usern = round(len(lines) / 15)
    rowlist = []
    for i in range(usern):
        if lines[15 * i][0] == '#':
            continue
        user_id = lines[15 * i].strip()
        followern = lines[15 * i + 4].strip()
        friendn = lines[15 * i + 7].strip()
        dict = {
            'uid': int(user_id),
            'followern': int(followern),
            'friendn': int(friendn),
        }
        rowlist.append(dict)

    df = pd.DataFrame(rowlist)
    df.to_csv('../data/user_basic.csv', index=False)


def create_user_detail(diffusion_file, detail_file):
    diffusion = pd.read_csv('../data/micro/{}.csv'.format(diffusion_file),
                            parse_dates=['time'], infer_datetime_format=True)
    diffusion.columns = ['original_status_id', 'original_user_id', 'time', 'user_id']

    diffusion.drop_duplicates(subset=['original_status_id', 'user_id'], inplace=True)
    grouped = diffusion.groupby(['original_user_id', 'user_id'])
    count = grouped.size()
    tm = diffusion.time.max() + timedelta(seconds=1)
    tf = grouped.time.min()
    rate = count.div(tf.apply(lambda x: (tm - x).total_seconds()))
    retweet = pd.DataFrame({
        'rate': rate,
        'count': count,
    }, index=count.index)
    # retweet.loc[np.isinf(retweet['rate']), 'rate'] = 0.0  # avoid divide by 0

    g = diffusion.groupby('user_id')
    outflow = g.size().div(g.time.min().apply(lambda x: (tm - x).total_seconds()))  # series

    # outflow.loc[np.isinf(outflow)] = 0.0
    outflow.name = 'outflow'

    A = retweet.reset_index()
    B = pd.DataFrame(outflow).reset_index()
    tmp = A.merge(B, left_on='original_user_id', right_on='user_id', how='inner')
    tmp.drop('user_id_y', axis=1, inplace=True)
    tmp.columns = ['original_user_id', 'user_id', 'count', 'rate', 'inflow']

    tmp['cxo'] = tmp['count'] * tmp['inflow']
    avg_inflow = tmp.groupby('user_id').aggregate({
        'cxo': np.sum,
        'count': np.sum,
        'inflow': np.sum,
    })

    avg_inflow['avg_inflow'] = avg_inflow['cxo'] / avg_inflow['count']
    A['cxr'] = A['count'] * A['rate']
    avg_outflow = A.groupby('original_user_id').aggregate({
        'cxr': np.sum,
        'count': np.sum,
    })
    avg_outflow['avg_outflow'] = avg_outflow['cxr'] / avg_outflow['count']
    user_detail = avg_inflow[['inflow', 'avg_inflow']].join(avg_outflow[['avg_outflow']], how='outer', )

    # user_detail.fillna(0, inplace=True)
    user_detail.dropna(inplace=True)
    user_detail = user_detail.join(outflow, how='inner')
    user_detail.reset_index(inplace=True)
    user_detail.columns = ['user_id', 'inflow', 'avg_inflow', 'avg_outflow', 'outflow']
    user_detail.to_csv('../data/micro/{}.csv'.format(detail_file), index=False)
    return


def create_user_dict():
    umap = open('../data/micro/uidlist.txt', 'r')
    linen = 0
    user_dict = {}
    for line in umap:
        uid = int(line.strip())
        user_dict[uid] = linen
        linen += 1
    umap.close()
    f = open('../data/micro/udict', 'wb')
    dump(user_dict, f)
    f.close()
    return


def map_index(df, col_list, dict_list):  # col is string array of column names, dict is a dictionary array
    df_mapped = df.copy()
    for i in range(len(col_list)):
        col = col_list[i]
        d = dict_list[i]
        # only apply if x exists as key , else return None

        df_mapped[col] = df_mapped[col].apply(lambda x: d.get(x))
        df_mapped = df_mapped.dropna(subset=[col, ])
    return df_mapped.set_index(col_list)


def create_user_data(basic_file, detail_file, user_data_file, suffix):
    user_basic = pd.read_csv('../data/micro/{}.csv'.format(basic_file))  # all users 1681085
    user_basic.drop_duplicates(subset=['uid'], inplace=True)  # 1655678
    with open('../data/micro/udict', 'rb') as f:
        udict = load(f)
    user_basic_mapped = map_index(user_basic, ['uid'], [udict])
    user_basic_mapped.reset_index(inplace=True)
    user_detail = pd.read_csv('../data/micro/{}.csv'.format(detail_file))  # users in diffusion 693665
    user_data = pd.merge(user_basic_mapped, user_detail, left_on='uid', right_on='user_id', how='inner')
    user_data = user_data[user_data['friendn'] != 0]  # remove zero entries
    user_data.drop('user_id', axis=1, inplace=True)
    user_data.reset_index(drop=True, inplace=True)
    user_data.fillna(0, inplace=True)
    # a smaller user dictionary for all users that appeared
    user_dict = {}
    for row in user_data[['uid']].itertuples():
        user_dict[row.uid] = row.Index
    with open('../data/micro/user_dict-{}'.format(suffix), 'wb') as f:
        dump(user_dict, f)
    # with open('../data/micro/userd-{}'.format(suffix),'rb') as f:
    #     user_dict = load(f)
    # user_mapped = map_index(user_data,['uid'],[user_dict]) #uid index
    # user_mapped.reset_index(inplace=True)
    user_mapped = user_data.drop('uid', axis=1).reset_index()
    user_mapped.columns = ['user_id', 'followern', 'friendn', 'inflow', 'avg_inflow', 'avg_outflow', 'outflow']
    user_mapped.to_csv('../data/micro/{}.csv'.format(user_data_file), index=False)
    return


# create_user_detail('diffusion-1k','user_detail-1k')
# create_user_data('user_basic', 'user_detail-1k', 'user_data-1k', '1k')
