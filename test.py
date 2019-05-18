import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from random import randint, uniform


# randomly generates some testing data


# list of random numbers
# [uniform(0,10) for i in range(100)]

# generate a list of random times
# [datetime(2016,8,8)+timedelta(seconds=uniform(0,delta)) for i in range(l)]

def generate_training():  #randomly generate a training set
    user_data = pd.DataFrame({
        'user_id': np.arange(100),
        'friend_count': np.random.randint(20, 1000, 100),
        'follower_count': np.random.randint(10, 1000, 100),
        'inflow': np.random.rand(100) * 50,
        'avg_inflow': np.random.rand(100) * 70,
        'outflow': np.random.rand(100) * 20,
    })

    # time delta should all > 1
    cascade = pd.DataFrame({
        'original_status_id': [1, ] * 200 + [2, ] * 100,
        'original_user_id': np.random.randint(0, 100, 300),
        'user_id': np.random.randint(0, 100, 300),
        'time': [datetime(2016, 8, 8) + timedelta(seconds=randint(1, 3 * 3600)) for i in range(300)]
    })
    return [user_data, cascade]


def generate_observed():
    # partial cascade
    pcascade = pd.DataFrame({
        'original_status_id': [3, ] * 100 + [4, ] * 100,
        'original_user_id': np.random.randint(0, 100, 200),
        'user_id': np.random.randint(0, 100, 200),
        'time': [datetime(2016, 8, 10) + timedelta(seconds=randint(1, 3 * 3600)) for i in range(200)]
    })

    # full cascade
    final = pd.DataFrame({
        'original_status_id': [3, 4],
        'final_count': [300, 400],
    })
    return [pcascade, final]
