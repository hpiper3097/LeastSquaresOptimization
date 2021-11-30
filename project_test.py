import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot, lag_plot

fname = './../data/data_filled/FILLED_log_2021_02_27-07_34_43_BB.csv'

df = pd.read_csv(fname)
objdim = df.count(axis=0).size  # expecting 3: [index, datetime, value]

#print(df.values)    #all the values in the dataframe [index, datetime, val we want to predict]
#print(df.count(axis=1).size)    #'size' of the columns axis (or number of rows..)

def split_list(list, window_len, stride=None):
    stride = window_len if stride is None else stride
    return [list[i:i+window_len] for i in range(0, len(list), stride) if len(list[i:i+window_len]) == window_len]

# A = [1, wi[:-1] .... ] where wi = sum(xi*w[-i+1]+ran(t), i in range(1, L))
#   w[-1] = x(1)*w[-2] + x(2)*w[-3] + ... + x(L-1)*w[0]
for i in range(1,20):
    print("ITERATION", i)
    print("***************************")
    dfWin = split_list(df, i*750)

    tempA = np.array([dfWin[k]['BtoB'].values[:-1] for k in range(0, len(dfWin))])
    A = np.insert(tempA, 0, np.ones(len(dfWin), dtype='uint64'), axis=1)

    B = [dfWin[k]['BtoB'].values[-1] for k in range(0, len(dfWin))]

    U, S, VT = np.linalg.svd(A, full_matrices=False)
    x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ B

    mse = np.array([((B[k] - np.sum(A[k])*x[k])**2)**(1/2) for k in range(len(B))])

    for e in mse:
        print(e)

    print()
    print("stdev of mse =", mse.std())
    print("mean of mse =", mse.mean())
    print()
    print("***************************")

    #pseudoinverse of A := pA minimizes ||Ax - b||**2, for x = pA*b
    #ignore singular values below certain threshold