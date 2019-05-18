import numpy as np
import pandas as pd
from datetime import date, timedelta
from math import log, exp
from scipy.optimize import minimize
from sklearn import linear_model

# testing
from test import generate_training, generate_observed


def get_delta_series(group):
    # return Series
    s = pd.Series((group['time'] - group['time'].min()).apply(lambda dt: dt / timedelta(seconds=1)))
    return s


def G1(Lambda, K, C, m, cascaden):
    logL = 0
    for i in cascaden:
        p = C.loc[i, :].pow(K).sum(level='original_user_id')  # index original_user_id
        logL = logL + \
               m.loc[i, :].mul(K.apply(log)).sum() + \
               C.loc[i, :].apply(log).sum(level='original_user_id').mul(K.sub(1)).sum() - \
               m.loc[i, :].mul(K).mul(Lambda.apply(log)).sum() - \
               Lambda.pow(K.mul(-1)).mul(p).sum()
    return -logL


def G2(Lambda, Beta, user_data, Alpha, usern):
    loss = Lambda.apply(log).sub(user_data.applymap(log).dot(Beta))
    return 1 / (2 * usern) * loss.apply(lambda x: x * x).sum() + Alpha[0] * Beta.abs().sum()  # L1 norm


def G3(K, Gamma, user_data, Alpha, usern):
    loss = K.apply(log).sub(user_data.applymap(log).dot(Gamma))
    return 1 / (2 * usern) * loss.apply(lambda x: x * x).sum() + Alpha[1] * Gamma.abs().sum()  # L1 norm


def cost(cascaden, usern, C, m, user_data, Alpha, Mu, Yita, Lambda, K, Beta, Gamma):
    J = G1(Lambda, K, C, m, cascaden) + Mu * G2(Lambda, Beta, user_data, Alpha, usern) + Yita * G3(K, Gamma, user_data,
                                                                                                   Alpha, usern)
    return J


def costLambda(Ldata, cascaden, usern, C, m, user_data, Alpha, Mu, K, Beta, Lindex):
    Lseries = pd.Series(Ldata, index=Lindex)
    J = G1(Lseries, K, C, m, cascaden) + Mu * G2(Lseries, Beta, user_data, Alpha, usern)
    return J


def derLambda(Ldata, cascaden, usern, C, m, user_data, Alpha, Mu, K, Beta, Lindex):
    derG1 = pd.Series(np.zeros(Ldata.shape), index=Lindex)
    Lseries = pd.Series(Ldata, index=Lindex)
    for i in cascaden:
        p = C.loc[i, :].pow(K).sum(level='original_user_id')
        a = m.loc[i, :].mul(K).div(Lseries)
        a.index = a.index.droplevel('original_status_id')
        b = Lseries.pow(K.sub(-1)).mul(p).mul(K).fillna(0)  # NAs appear
        derG1 = derG1.add(a).add(b)
    loss = Lseries.apply(log).sub(user_data.applymap(log).dot(Beta))
    derG2 = loss.div(Lseries).div(usern)
    return derG1.add(derG2.mul(Mu))


def costK(K, cascaden, usern, C, m, user_data, Alpha, Yita, Lambda, Gamma, Kindex):
    Kseries = pd.Series(K, index=Kindex)
    J = G1(Lambda, Kseries, C, m, cascaden) + Yita * G3(Kseries, Gamma, user_data, Alpha, usern)
    return J


def derK(K, cascaden, usern, C, m, user_data, Alpha, Yita, Lambda, Gamma, Kindex):
    Kseries = pd.Series(K, index=Kindex)
    derG1 = pd.Series(np.zeros(K.shape), index=Kindex)
    for i in cascaden:
        tmpC = C.loc[i, :]
        tmpC.index = tmpC.index.droplevel('original_status_id')
        tmpm = m.loc[i, :]
        tmpm.index = tmpm.index.droplevel('original_status_id')
        p = tmpC.pow(Kseries).sum(level='original_user_id')
        q = tmpC.pow(Kseries).mul(tmpC.apply(log)).sum(level='original_user_id')
        derG1 = derG1.add(tmpm.div(Kseries)) \
            .add(tmpC.apply(log).sum(level='original_user_id')) \
            .sub(tmpm.mul(Lambda.apply(log))) \
            .add(Lambda.pow(Kseries.mul(-1)).mul(Kseries.apply(log)).mul(p)) \
            .sub(Lambda.pow(Kseries.mul(-1)).mul(q)).fillna(0)
    loss = Kseries.apply(log).sub(user_data.applymap(log).dot(Gamma))
    derG3 = loss.div(Kseries).div(usern)
    return derG1.add(derG3.mul(Yita))


def maxlikelihood(cascaden, usern, C, m, user_data, Alpha, Mu, Yita, Lambda, K, Beta, Gamma, iters):
    Lindex = Lambda.index
    Kindex = K.index
    for it in range(iters):
        # minimize J w.r.t Lambda ,Newton method
        params1 = (cascaden, usern, C, m, user_data, Alpha, Mu, K, Beta, Lindex)
        bnds1 = ((0, None),) * Lambda.shape[0]  # Lambda>0
        res = minimize(costLambda, Lambda.data, params1, jac=derLambda, options={'maxiter': 50})
        Lambda = pd.Series(res.x, Lindex)
        # minimize J w.r.t. K, Newton method
        params2 = (cascaden, usern, C, m, user_data, Alpha, Yita, Lambda, Gamma, Kindex)
        bnds2 = ((0, None),) * K.shape[0]  # K>0
        res = minimize(costK, K.data, params2, jac=derK, options={'maxiter': 50})
        K = pd.Series(res.x, Kindex)
        # minimize G2 w.r.t Beta LASSO
        clf = linear_model.Lasso(alpha=Alpha[0], max_iter=100, fit_intercept=False)
        clf.fit(user_data.applymap(log), Lambda.apply(log))
        Beta = pd.Series(clf.coef_, index=['avg_inflow', 'follower_count', 'friend_count', 'inflow', 'outflow'])
        # minimize G3 w.r.t Gamma LASSO
        clf = linear_model.Lasso(alpha=Alpha[1], max_iter=100, fit_intercept=False)
        clf.fit(user_data.applymap(log), K.apply(log))
        Gamma = pd.Series(clf.coef_, index=['avg_inflow', 'follower_count', 'friend_count', 'inflow', 'outflow'])
    return [Lambda, K, Beta, Gamma]


def survival(t, Lambda, K):
    S = exp(-(t / Lambda) ** K)
    return S


# prediction
# partial cascade
# cascade_id | original_user_id | user_id | time


def predict(pcascade, user_data, Beta, Gamma, usern):
    cascade_set = pcascade.original_status_id.unique()
    pcascade.sort_values(['original_status_id', 'original_user_id', 'time'], axis=0, inplace=True)
    p = pcascade.set_index(['original_status_id', 'original_user_id'])
    p.drop('user_id', axis=1, inplace=True)  # p is a DataFrame
    replynum = pcascade.groupby(['original_status_id', 'original_user_id']).size()
    final_time = timedelta(days=10).total_seconds()
    prediction = pd.Series(np.ones(cascade_set.shape[0]), index=cascade_set)

    user_lambda = user_data.applymap(log).dot(Beta).apply(exp)
    user_k = user_data.applymap(log).dot(Gamma).apply(exp)
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
    final.set_index('original_status_id', inplace=True)
    fseries = pd.Series(final['final_count'])
    abserror = prediction.sub(fseries).abs().div(fseries)
    return abserror


def main(test=False, user_data_file=None, cascade_file=None, pcascade_file=None, final_file=None, prediction_file=None):
    # user data
    # user_id | friend_count | follower_count | inflow | avg_inflow | outflow
    # cascade data
    # original_status_id (cascade_id) | original_user_id | user_id | time
    if not test:
        user_data = pd.read_csv(user_data_file)  # DataFrame
        cascade = pd.read_csv(cascade_file)  # DataFrame
        pcascade = pd.read_csv(pcascade_file)  # DataFrame
        final = pd.read_csv(final_file)  # DataFrame
    else:
        [user_data, cascade] = generate_training()
        [pcascade, final] = generate_observed()

    user_data.set_index('user_id', inplace=True)
    user_data.sort_index(axis=0, inplace=True)
    cascade.sort_values(['original_status_id', 'original_user_id', 'time'], axis=0, inplace=True)
    # M is the number of reposts for each <cascade, original_user> pair
    m = cascade.groupby(['original_status_id', 'original_user_id']).size()
    C = cascade.groupby(['original_status_id', 'original_user_id']).apply(get_delta_series)
    C.index = C.index.droplevel(2)  # drop the original index
    C = C[C != 0]
    # C.set_index(['cascade_id','original_user_id'],inplace=True)
    # parameters
    usern = user_data.shape[0]
    cascaden = cascade.original_status_id.unique()  # series of unique cascade ids
    K = pd.Series(np.ones(usern), index=user_data.index)
    Lambda = pd.Series(np.ones(usern), index=user_data.index)
    K.index.name = 'original_user_id'
    Lambda.index.name = 'original_user_id'
    Beta = pd.Series(np.random.rand(5), index=['avg_inflow', 'follower_count', 'friend_count', 'inflow', 'outflow'])
    Gamma = pd.Series(np.random.rand(5), index=['avg_inflow', 'follower_count', 'friend_count', 'inflow', 'outflow'])
    Alpha = [6 * 10 ** (-5), 8 * 10 ** (-6)]  # according to the paper
    Mu = 10
    Yita = 10

    input("Finished preparing data and parameters.")

    # der1 = derLambda(Lambda.data, cascaden, usern, C, m, user_data, Alpha, Mu, K, Beta, Lambda.index)
    # input("computed derivative of lambda, shape {}".format(der1.shape))
    #
    # der2 = derK(K.data, cascaden, usern, C, m, user_data, Alpha, Yita, Lambda, Gamma, K.index)
    # input("computed derivative of K, shape {}".format(der2.shape))

    [Lambda, K, Beta, Gamma] = maxlikelihood(cascaden, usern, C, m, user_data, Alpha, Mu, Yita, Lambda, K, Beta, Gamma,
                                             10)
    input("Finished learning parameters.")

    prediction = predict(pcascade, user_data, Beta, Gamma, usern)
    if prediction_file:
        prediction.to_csv(prediction_file)
    input('Finished prediction.')

    error = accuracy(prediction, final)
    input('Prediction accuracy is {}'.format(error))

    return


main(test=True)
