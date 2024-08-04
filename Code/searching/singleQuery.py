import math
import numpy as np
from typing import List, Tuple
import numpy as np
import numba as nb
from numba import njit
# import tensorflow as tf
import random
from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# from keras import backend as K
import time
metric = "DTW"
th=0.001
method = "PSS"
dataset = "Xi'an"
dataDistribution = {
    "Xi'an": {
        "lat": (34.20, 34.29),
        "lon": (108.91, 109.00)
    },
    "Chengdu": {
        "lat": (30.65, 30.73),
        "lon": (104.04, 104.13)
    },
    "Porto": {
        "lat": (41.11, 41.19),
        "lon": (-8.67, -8.57)
    }
}
centerPoint: Tuple[float, float] = (
    sum(dataDistribution[dataset]["lat"]) / 2, 
    sum(dataDistribution[dataset]["lon"]) / 2
)




def queryForSubtrajectory(query, data, method, met,city):
    global metric
    global th
    met = met.upper()
    metric = met

    if city=='porto' and met=='EDR':
        th=0.003

    if method != "ExactS":
        return heuristic(data, query, method)
    elif method == "ExactS":
        return ExactS(np.array(data), np.array(query))


def str2path(string: str) -> List[Tuple[float, float]]:
    return [(ord(i) - ord('a'), ord(i) - ord('a')) for i in string]


class Distance:
    def __init__(self, N, M): # N = length of C, M = length of Q
        self.D0 = np.zeros((N + 1, M + 1))
        self.flag = np.zeros((N, M))
        self.D0[0,1:] = np.inf
        self.D0[1:,0] = np.inf
        self.D = self.D0[1:,1:] #shallow copy!!
        #print(self.D)
    #@jit
    def DTW(self, traj_C, traj_Q):
        # traj_C = np.array(traj_C)
        # traj_Q = np.array(traj_Q)
        n = len(traj_C)
        m = len(traj_Q)
        for i in range(n):
            for j in range(m):
                if self.flag[i,j] == 0:
                    cost =  np.linalg.norm(traj_C[i] - traj_Q[j]) if metric != "EDR" else (
                        0 if np.linalg.norm(traj_C[i] - traj_Q[j]) < 0.001 else 1)
                    self.D[i,j] = (cost + min(self.D0[i,j],self.D0[i,j+1],self.D0[i+1,j])) if metric == "DTW" else \
                        min(self.D0[i,j] + cost, self.D0[i,j+1] + 1, self.D0[i+1,j] + 1)
                    self.flag[i,j] = 1
                    #print(self.D)
        return self.D[n-1, m-1]



@njit
def E_Distance(N, M, traj_C, traj_Q):
    global metric
    global th
    D0 = np.zeros((N + 1, M + 1))
    flag = np.zeros((N, M))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D = D0[1:, 1:]  # shallow copy!!

    for i in range(len(traj_C)):

        for j in range(len(traj_Q)):
            if flag[i, j] == 0:
                cost = np.linalg.norm(traj_C[i] - traj_Q[j]) if metric != "EDR" else (
                    0 if np.linalg.norm(traj_C[i] - traj_Q[j]) < th else 1)
                D[i, j] = (cost + min(D0[i, j], D0[i, j + 1], D0[i + 1, j])) if metric == "DTW" else \
                    min(D0[i, j] + cost, D0[i, j + 1] + 1, D0[i + 1, j] + 1)
                flag[i, j] = 1
                # print(D)
    return D[len(traj_C) - 1, len(traj_Q) - 1]

@nb.jit(nopython=True)
def ExactS(traj_c, traj_q):


    subsim = 999999
    subtraj = [0, len(traj_c)-1]
    N = len(traj_c)
    for i in range(N):
        N1, M1 = len(traj_c[i:]), len(traj_q)
        for j in range(i, N):
            temp=E_Distance(N1, M1 ,traj_c[i:j+1], traj_q)
            if temp < subsim:
                subsim = temp
                subtraj = [i, j]

    return subsim, subtraj



def heuristic_suffix_opt(traj_c, traj_q, index, opt, N, M):
    if index == len(traj_c):
        return 999999
    if opt == 'POS' or opt == 'POS-D':
        return 999999
    if opt == 'PSS':
        return E_Distance(N, M,traj_c[index:][::-1], traj_q[::-1])

def heuristic(traj_c, traj_q, opt, delay_K=5):
    '''

    :param traj_c:
    :param traj_q:
    :param opt: PSS, POS, POS-D, RLS, RLS-Skip, SizeS
    :param delay_K:
    :return:
    '''
    delay = 0
    subsim = 999999
    subtraj = [0, len(traj_c) - 1]
    split_point = 0
    N, M =  len(traj_c), len(traj_q)
    N2, M2 =  len(traj_c), len(traj_q)
    pos_d_coll = []
    pos_d_f = False
    temp = 'non'



    if opt != 'POS-D':
        for i in range(len(traj_c)):
            # submit prefix
            presim = E_Distance(N, M,traj_c[split_point:i + 1], traj_q)
            sufsim = heuristic_suffix_opt(traj_c, traj_q, i + 1, opt,N2, M2)

            if presim < subsim or sufsim < subsim:
                temp = i + 1
                subsim = min(presim, sufsim)
                if presim < sufsim:
                    subtraj = [split_point, (temp - 1)]
                else:
                    subtraj = [temp, len(traj_c) - 1]
                split_point = temp
                N, M = len(traj_c[i:]), len(traj_q)

    # print(subsim)
    return subsim, subtraj







def set_metric(x):
    global metric
    metric=x




