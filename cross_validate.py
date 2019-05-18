import pandas as pd
import numpy as np
from predict import create_pcascade, create_final, predict, accuracy


# TODO:this file needs revision


def cross_validate(cascade_file, foldn):  # divide a large cascade file into n folds
    cascade = pd.read_csv(cascade_file, parse_dates=['time'],
                          infer_datetime_format=True)
    cascade.columns = ['original_status_id', 'original_user_id', 'time', 'user_id']
    cascade_list = cascade.original_status_id.unique()
    per = np.random.permutation(cascade_list)
    chunksize = round(cascade_list.shape[0] / foldn)
    abserror = 0
    for i in range(foldn):
        test_indexes = per[i * chunksize:(i + 1) * chunksize]
        testset = cascade.take(test_indexes)
        train_indexes = np.concatenate(per[:i * chunksize], per[(i + 1) * chunksize:])
        trainset = cascade.take(train_indexes)
        [Lambda, K, Gamma, Beta] = train(testset)
        pcascade = create_pcascade(trainset, s, k)
        prediction = predict(pcascade, user_matrix, Beta, Gamma, usern)
        final = create_final(trainset)
        abserror += accuracy(prediction, final)
        # testset.to_csv('../data/micro/testset-{}'.format(i))
    meanerror = abserror / foldn
    print(meanerror)
    return
