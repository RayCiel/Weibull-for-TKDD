import pandas as pd
import numpy as np
from scipy import sparse
from scipy.optimize import minimize,check_grad
from sklearn import linear_model
from datetime import timedelta
from pickle import load, dump
from math import log, pow
from utility import to_sparse,map_index

def get_delta_series(group):
    # return Series
    s = pd.Series((group['time'] - group['time'].min()).apply(lambda dt: dt / timedelta(seconds=1)))
    return s


def get_delta_df(group):
    return pd.DataFrame({
            'delta':(group['time'] - group['time'].min()).apply(lambda dt: dt / timedelta(minutes=1)),
            'original_status_id':group['original_status_id'],
            'original_user_id':group['original_user_id'],
            #'index':np.arange(len(group.index)),
        })


def prepare(user_data, cascade_mapped, messagen, usern):
    user_data.set_index('user_id', inplace=True)
    user_matrix = user_data.as_matrix()
    cascade = cascade_mapped.reset_index().dropna()
    # cascade.sort_values(['original_status_id', 'original_user_id', 'time'], axis=0, inplace=True)
    # M is the number of reposts for each <cascade, original_user> pair
    m = cascade.groupby(['original_status_id', 'original_user_id']).size()

    C = cascade.groupby(['original_status_id', 'original_user_id']).apply(get_delta_df)  # DataFrame
    C.original_user_id = C.original_user_id.astype('int64')
    C = C[C['delta'] != 0]

    return [user_matrix, m, C]


def get_C_row(C, usern, coln, i):
    tmp = pd.DataFrame(C.loc[i, :]).reset_index().drop('original_status_id', axis=1).set_index(['original_user_id'],
                                                                                               append=True)
    tmp.index = tmp.index.swaplevel(i=0, j=1)
    c_sparse = to_sparse(tmp['time'], usern, coln)
    return c_sparse.toarray()


def to_col(l):
    return np.ndarray(buffer=l, shape=(l.shape[0], 1))


def G1(Lambda,K,t1,t2,cascaden):
    logL =0
    for i in cascaden:
        if len(t1.loc[i].shape)==1:
            u = t1.loc[i].original_user_id
            time = t2.loc[i].loc[u].delta
            userK = K[u]
            userLambda = Lambda[u]
            logL += log(userK)+log(time)*(userK-1)-userK*log(userLambda)-pow(userLambda,userK*(-1))*pow(time,userK)
        else:
            users = t1.loc[i].original_user_id.drop_duplicates().as_matrix().tolist()
            for u in users:
                time = t2.loc[i].loc[u].delta.as_matrix()
                userK = K[u]
                userLambda = Lambda[u]
                m = time.shape[0]
                a = m * log(userK)
                b = np.log(time).sum()*(userK-1)
                c = m*userK*log(userLambda)
                d = pow(userLambda,userK*(-1))*np.power(time,userK).sum()
                logL += a+b-c-d
    return -logL


def G2(Lambda, Beta, user_data, Alpha, usern):
    loss = np.log(Lambda) - (np.log(user_data).dot(Beta))  # user_data should have no zeroes
    return 1 / (2 * usern) * np.linalg.norm(loss) ** 2 + Alpha[0] * np.abs(Beta).sum()  # L1 norm


def G3(K, Gamma, user_data, Alpha, usern):
    loss = np.log(K) - (np.log(user_data).dot(Gamma))
    return 1 / (2 * usern) * np.linalg.norm(loss)**2 + Alpha[1] * np.abs(Gamma).sum()  # L1 norm


def cost(cascaden, usern, t1,t2, m, user_data, Alpha, Mu, Yita, Lambda, K, Beta, Gamma):
    J = G1(Lambda, K, t1,t2,cascaden) + Mu * G2(Lambda, Beta, user_data, Alpha, usern) + Yita * G3(K, Gamma,
                                                                                                          user_data,
                                                                                                          Alpha, usern)
    return J


def costLambda(Lambda, cascaden, usern, t1,t2, user_data, Alpha, Mu, K, Beta):
    J = G1(Lambda, K, t1,t2,cascaden) + Mu * G2(Lambda, Beta, user_data, Alpha, usern)
    return J


def derLambda(Lambda, cascaden, usern, t1,t2,user_data, Alpha, Mu, K, Beta):
    derG1 = np.zeros(Lambda.shape)  # col vector
    for i in cascaden:
        if len(t1.loc[i].shape) == 1:
            u = t1.loc[i].original_user_id
            time = t2.loc[i].loc[u].delta
            userK = K[u]
            userLambda = Lambda[u]
            p = pow(time, userK)
            a = userK / userLambda
            b = pow(userLambda, userK*(-1) - 1) * p * userK
            derG1[u] += a - b
        else:
            users = t1.loc[i].original_user_id.drop_duplicates().as_matrix().tolist()
            for u in users:
                time = t2.loc[i].loc[u].delta.as_matrix()
                userK = K[u]
                userLambda = Lambda[u]
                m = time.shape[0]
                p = np.power(time,userK).sum()
                a = m*userK/userLambda
                b = pow(userLambda,userK*(-1)-1) *p *userK
                derG1[u] += a-b

    loss = np.log(Lambda) - np.log(user_data).dot(Beta).reshape(Lambda.shape)
    derG2 = 1 / usern * (loss / Lambda)
    result = derG1 + derG2 * Mu
    return result


def costK(K, cascaden, usern, t1,t2,user_data, Alpha, Yita, Lambda, Gamma):
    J = G1(Lambda, K, t1,t2, cascaden) + Yita * G3(K, Gamma, user_data, Alpha, usern)
    return J


def derK(K, cascaden, usern, t1,t2,user_data, Alpha, Yita, Lambda, Gamma):
    derG1 = np.zeros(K.shape)
    for i in cascaden:
        if len(t1.loc[i].shape) == 1:
            u = t1.loc[i].original_user_id
            time = t2.loc[i].loc[u].delta
            userK = K[u]
            userLambda = Lambda[u]
            p = pow(time, userK)
            q = pow(time, userK) * log(time)
            derG1[u] += - 1/ userK - log(time) + log(userLambda) - pow(userLambda, userK * (-1)) * log(userK) * p \
                + pow(userLambda, userK * (-1)) * q

    else:
            users = t1.loc[i].original_user_id.drop_duplicates().as_matrix().tolist()
            for u in users:
                time = t2.loc[i].loc[u].delta.as_matrix()
                userK = K[u]
                userLambda = Lambda[u]
                m = time.shape[0]
                p = np.power(time,userK).sum()
                q = (np.power(time,userK) * np.log(time)).sum()
                derG1[u] += - m/userK - np.log(time).sum() + m*log(userLambda) - pow(userLambda,userK*(-1)) *log(userLambda)*p\
                + pow(userLambda,userK*(-1))*q

    loss = np.log(K) - np.log(user_data).dot(Gamma).reshape(K.shape)
    derG3 = 1 / usern * (loss / K)
    result = derG1 + derG3 * Yita
    return result


def maxlikelihood(cascaden, usern, C, m, user_data, Alpha, Mu, Yita, Lambda, K, Beta, Gamma, iters):
    t1 = C.set_index('original_status_id')
    t2 = C.set_index(['original_status_id', 'original_user_id'])
    for it in range(iters):
        print(it)
        # minimize J w.r.t Lambda ,Newton method
        params1 = (cascaden, usern, t1,t2, user_data, Alpha, Mu, K, Beta)
        bnds1 = ((1, None),) * Lambda.shape[0]  # Lambda>0
        res = minimize(costLambda, Lambda, params1,jac=derLambda,options={'maxiter': 30},bounds=bnds1)
        Lambda = res.x
        print("Lambda optimized")
        # minimize J w.r.t. K, Newton method
        params2 = (cascaden, usern, t1,t2,user_data, Alpha, Yita, Lambda, Gamma)
        bnds2 = ((1, None),) * K.shape[0]  # K>0
        res = minimize(costK, K, params2, jac=derK,options={'maxiter': 30},bounds=bnds2)
        K = res.x
        print("K optimized")
        # minimize G2 w.r.t Beta LASSO
        clf = linear_model.Lasso(alpha=Alpha[0], max_iter=50, fit_intercept=False)
        clf.fit(np.log(user_data), np.log(Lambda))
        Beta = clf.coef_
        print("Beta optimized")
        # minimize G3 w.r.t Gamma LASSO
        clf = linear_model.Lasso(alpha=Alpha[1], max_iter=50, fit_intercept=False)
        clf.fit(np.log(user_data), np.log(K))
        Gamma = clf.coef_
        print("Gamma optimized")
    return [Lambda, K, Beta, Gamma]


def main(user_data_file, cascade_file, messaged_name, userd_name, suffix, pcascade_file=None, final_file=None, ):
    # user_id| followern | friendn | inflow | avg_inflow | avg_outflow | outflow
    user_data = pd.read_csv(user_data_file)
    cascade = pd.read_csv(cascade_file,
                          parse_dates=['time'],
                          infer_datetime_format=True)
    cascade.columns = ['original_status_id', 'original_user_id', 'time', 'user_id']  # DataFrame
    if pcascade_file:
        pcascade = pd.read_csv(pcascade_file,
                               parse_dates=['time'],
                               infer_datetime_format=True)  # DataFrame
        pcascade.columns = ['original_status_id', 'original_user_id', 'time', 'user_id']
    if final_file:
        final = pd.read_csv(final_file)  # DataFrame
    with  open('../data/micro/{}'.format(messaged_name), 'rb') as mfile:
        message_dict = load(mfile)
    with open('../data/micro/{}'.format(userd_name), 'rb') as ufile:
        user_dict = load(ufile)
    cascade_mapped = map_index(cascade, ['original_status_id', 'original_user_id'], [message_dict, user_dict])
    cascade_list = cascade_mapped.reset_index().original_status_id.unique()
    messagen = len(message_dict)
    usern = len(user_dict)

    [user_matrix, m, C] = prepare(user_data, cascade_mapped, messagen, usern)

    K = np.random.rand(usern, 1) + 1  # np.ones((usern, 1))
    Lambda = np.random.rand(usern, 1) + 1
    Beta = np.random.rand(6, 1)
    Gamma = np.random.rand(6,1)
    Alpha = [6 * 10 ** (-5), 8 * 10 ** (-6)]  # according to the paper
    Mu = 10
    Yita = 10

    input("Finished preparing data and parameters.")
    [Lambda, K, Beta, Gamma] = maxlikelihood(cascade_list, usern, C, m,
                                             user_matrix, Alpha, Mu, Yita, Lambda, K, Beta, Gamma, 10)
    # g1 = G1(Lambda, K, C, m_sparse, messagen, usern)
    # g2 = G2(Lambda, Beta, user_matrix, Alpha, usern)
    # g3 = G3(K, Gamma, user_matrix, Alpha, usern)


    # dl = derLambda(Lambda, cascade_list, usern, C, m_sparse, user_matrix, Alpha, Mu, K, Beta)
    # dk = derK(K, messagen, usern, C, m_sparse, user_matrix, Alpha, Yita, Lambda, Gamma)
    input('training finished')
    with open('../data/micro/beta-{}'.format(suffix), 'wb') as f:
        dump(Beta, f)
    with open('../data/micro/gamma-{}'.format(suffix), 'wb') as f:
        dump(Gamma, f)

    return


main('../data/micro/user_data-1k.csv', '../data/micro/diffusion-1k.csv', 'messaged-1k', 'user_dict-1k', '1k')
