import pandas as pd
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
from datetime import timedelta


# loading data
# change the name of the cascade file
def get_delta_df(group):
    return pd.DataFrame({
        'time': (group['time'] - group['time'].min()).apply(lambda dt: dt / timedelta(minutes=1)),
        'original_status_id': group['original_status_id'],
        'original_user_id': group['original_user_id'],
        # 'index':np.arange(len(group.index)),
    })


# ug = wcascade.groupby('original_user_id')
# users = wcascade.original_user_id.unique()
# user0 = users[11]
# data = ug.get_group(user0)

# create partially observed cascade
# time is datetime object
def create_observation(cascade, time):
    return cascade[cascade['time'] < time.total_seconds() / 60]


# group by user
def fit_weibull(group):
    shape, loc, scale = weibull_min.fit(group['time'].values, floc=0)
    return pd.Series({
        'shape': shape,
        'scale': scale,
    })


def predict(group, params):
    current_size = group.shape[0]
    user = group.iloc[0].original_user_id
    shape = params.loc[user].shape
    scale = params.loc[user].scale
    predicted = group.shape[0] / weibull_min.cdf(group['time'].values.max(), shape, loc=0, scale=scale)[0]
    return pd.Series({
        'predicted': predicted,
    })


# for regression
def accuracyreg(prediction, final):
    results = prediction.join(final, how='inner')
    results['se'] = (results['predicted'] - results['size']) ** 2
    results['pe'] = (results['predicted'] - results['size']) / results['size']
    results['ape'] = abs(results['pe'])
    results['var'] = (results['size'] - results['size'].mean()) ** 2
    return results


# for classification
def accuracycla(prediction, final, threshold):
    final['popular'] = final['size'].apply(lambda x: x >= threshold)
    prediction['predicted'] = prediction.predicted.apply(lambda x: x >= threshold)
    results = prediction.join(final, how='inner')
    results['tp'] = prediction['predicted'] & final['popular']
    results['fp'] = prediction['predicted'] & ~final['popular']
    results['tn'] = ~prediction['predicted'] & ~final['popular']
    results['fn'] = ~prediction['predicted'] & final['popular']
    return results


def peek_predict(wcascade, threshold, timepts=[6, 12, 18, 24]):
    dlist = []

    final = pd.DataFrame(wcascade.groupby('original_status_id').size(), columns={'size', })
    resultsall = final.copy()

    for i in timepts:
        print(i)
        pcascade = create_observation(wcascade, timedelta(hours=i))
        weibull_params = pcascade.groupby('original_user_id').apply(fit_weibull)
        prediction = pcascade.groupby('original_status_id').apply(lambda x: predict(x, weibull_params))
        resultsall[str(i)] = prediction
        resultsreg = accuracyreg(prediction, final)
        resultscla = accuracycla(prediction, final, threshold)
        precision = resultscla.tp.sum() / (resultscla.tp.sum() + resultscla.fp.sum())
        recall = resultscla.tp.sum() / (resultscla.tp.sum() + resultscla.fn.sum())
        # extreme cases
        d = {
            'predicttime': i,
            'mse': resultsreg.se.mean(),
            'medianse': resultsreg.se.median(),
            'mape': resultsreg.ape.mean(),
            'medianape': resultsreg.ape.median(),
            'R2': 1 - resultsreg['se'].sum() / resultsreg['var'].sum(),
            'precision': precision,
            'recall': recall,
            'f1': 2 / (1 / precision + 1 / recall),
        }
        dlist.append(d)
    evaluation = pd.DataFrame(dlist)
    resultsall = resultsall.reset_index()
    return [evaluation, resultsall]


def main(cascade_file,timepts):
    # cascade_file = '../data/micro/diffusion-10k.csv'
    # cascade = pd.read_csv(cascade_file,
    #                      parse_dates=['time'],
    #                      infer_datetime_format=True)
    # cascade.columns = ['original_status_id', 'original_user_id', 'time', 'user_id']
    cascade_delta = pd.read_csv(cascade_file) #delta is float
    cascade_delta.rename(columns={'delta': 'time'}, inplace=True)
    # cascade_delta = cascade.groupby('original_status_id').apply(get_delta_df)
    threshold = cascade_delta.groupby('original_status_id').size().quantile(q=0.99, interpolation='nearest')
    # remove initial post times
    wcascade = cascade_delta[cascade_delta['time'] != 0]
    [evaluation, resultsall] = peek_predict(wcascade, threshold, timepts)
    evaluation.to_csv('evaluation-twitter-sample0.csv', index=False)
    resultsall.to_csv('prediction-twitter-sample0.csv', index=False)
    return


main('data0.csv',[i + 1 for i in range(24)])
