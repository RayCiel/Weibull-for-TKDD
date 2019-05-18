import pandas as pd
import numpy as np
from math import exp,log
from pickle import load,dump
from datetime import timedelta
from utility import map_index, create_user_dict, create_message_dict
from random import randint

def survival(t, Lambda, K):
    S = exp(-(t / Lambda) ** K)
    return S

# prediction
# partial cascade
# cascade_id | original_user_id | user_id | time


def create_pcascade(cascade,s,k):#create a partial cascade from full cascade
    ck = cascade.groupby(['original_status_id', 'original_user_id']).filter(lambda x: x.shape[0] > k)
    pcascade = ck.sort_values(['time']).groupby(['original_status_id','original_user_id']).head(s)
    final = ck.groupby(['original_status_id']).size()  # series
    return [pcascade, final]


def predict(pcascade, user_data, Beta, Gamma, usern):
    cascade_set = pcascade.original_status_id.unique()
    pcascade.sort_values(['original_status_id', 'original_user_id', 'time'], axis=0, inplace=True)
    p = pcascade.set_index(['original_status_id', 'original_user_id'])
    p.drop('user_id', axis=1, inplace=True)  # p is a DataFrame
    replynum = pcascade.groupby(['original_status_id', 'original_user_id']).size()
    final_time = timedelta(days=10).total_seconds() / 60  #total minutes
    prediction = pd.Series(np.ones(cascade_set.shape[0]), index=cascade_set)

    user_lambda = np.exp(np.log(user_data).dot(Beta))
    user_k = np.exp(np.log(user_data).dot(Gamma))
    for i in cascade_set:  # for every cascade
        limit = p.loc[i]['time'].max()  # current recorded time
        user_set = p.loc[i].index.unique()
        for user in user_set:  # for each user
            t = p.loc[i, user]['time'].min()
            deathrate = max(1 - survival((limit - t).total_seconds(), user_lambda[user], user_k[user]), 1 / usern)
            fdrate = max(1 - survival(final_time, user_lambda[user], user_k[user]), 1 / usern)
            prediction.loc[i] = prediction.loc[i] + (replynum.loc[i, user] * fdrate / deathrate)
            # round?
    return prediction


def accuracy(prediction, final):
    abserror = prediction.div(final)
    return abserror


def main(s, k, testsuffix, suffix, userd_name, prediction_file=None, final_file=None):
    cascade = pd.read_csv('../data/micro/diffusion-{}.csv'.format(testsuffix), parse_dates=['time'],
                          infer_datetime_format=True)
    cascade.columns = ['original_status_id', 'original_user_id', 'time', 'user_id']

    [pcascade, final] = create_pcascade(cascade, s, k)
    print('pcascade_created')
    message_dict = create_message_dict(pcascade)
    #user dict and user data is created for the entire cascade dataset
    with open('../data/micro/{}'.format(userd_name), 'rb') as ufile:
        user_dict = load(ufile)
    pcascade_mapped = map_index(pcascade, ['original_status_id', 'original_user_id'], [message_dict, user_dict])
    cascade_list = pcascade_mapped.reset_index().original_status_id.unique()
    messagen = len(message_dict)
    usern = len(user_dict)

    user_data = pd.read_csv('../data/micro/user_data-{}.csv'.format(suffix))
    user_data.set_index('user_id', inplace=True)
    user_matrix = user_data.as_matrix()
    with open('../data/micro/beta-{}'.format(suffix), 'rb') as f:
        Beta = load(f)
    with open('../data/micro/gamma-{}'.format(suffix), 'rb') as f:
        Gamma = load(f)
    print('begin prediction')
    prediction = predict(pcascade_mapped.reset_index(), user_matrix, Beta, Gamma, usern)
    if prediction_file:
        prediction.to_csv('../data/micro/{}.csv'.format(prediction_file))
    input('Finished prediction.')
    final_df = map_index(pd.DataFrame(final).reset_index(), ['original_status_id', ], [message_dict, ])
    if final_file:
        final_df.to_csv('../data/micro/{}.csv'.format(final_file))
    error = accuracy(prediction, final_df[0])
    input('Prediction accuracy is {}'.format(error))
    return


main(80, 100, '100', '1k', 'user_dict-1k', 'prediction-100-80')
