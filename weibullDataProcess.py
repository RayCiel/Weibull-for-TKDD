import numpy as np
import json
import time
from numpy import *
import numpy as np
import math
import scipy.stats as st
import csv


#original_mid original_uid original_time retweet_num
#N
#retweet_uid retweet_time retweet_mid
#retweet_text


retweet_num = []
retweet_N = []
retweet_id = []
num = 0
retweet_time = []

filein = '../../Data/retweetWithContent/repost0.txt'
#filein = 't.txt'
with open(filein, 'r', encoding = 'gbk') as fin:
    ce = fin.readlines()
    retweet_id = []
    retweet_time = []
    #print(retweet_time)
    #print(len(ce))
    i = 0
    while(i < len(ce)):
        ce[i] = ce[i].split()
        #print(ce[i])
        retweet_id.append([])
        original_mid = ce[i][0]
        original_uid=ce[i][1]
        original_time = (time.mktime(time.strptime(str(ce[i][2]), "%Y-%m-%d-%H:%M:%S")))

        i += 1
        N = int(ce[i])
        retweet_N.append(N)
        i += 1
        #print(str(i) + str(ce[i][2]))

        for j in range(N):
            #print(str(i) + str(ce[i]))

            ce[i] = ce[i].split()
    # cascade.columns = ['original_status_id', 'original_user_id', 'time', 'user_id']
            retweet_id[num].append(original_mid)
            retweet_id[num].append(original_uid)

            retweet_id[num].append(time.mktime(time.strptime(str(ce[i][1]), "%Y-%m-%d-%H:%M:%S")) - original_time)
            retweet_id[num].append(str(ce[i][0]))#uid
            i += 1

            i+=1
        num += 1
    print(len(retweet_time))
    print(len(original_uid))
    print(len(retweet_id))

dataout = 'data0.csv'
#indexout = 'index0.csv'

with open(dataout, 'w') as fout:
    fout.write("original_status_id" + ',' + "original_user_id" + ',' + 'time' + ',' + 'user_id' + '\n')
    for i in range(0, len(retweet_id)):
        for j in range(int(len(retweet_id[i]) / 4)):
            fout.write(str(retweet_id[i][j * 4]) + ',' + str(retweet_id[i][j * 4 + 1]) + ',' + str(retweet_id[i][j * 4 + 2]) + ',' + str(retweet_id[i][j * 4 + 3]) + '\n')
