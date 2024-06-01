import numpy as np
import itertools
from tqdm import tqdm
from time import time
import pdb
import matplotlib.pyplot as plt
import mne
import pickle
from scipy.stats import chi2
import sys
sys.path.append(".")
from utils import utils, correlation
from torus_graph_model.model import *
import json


def S1_j(x):
    '''
    x : d  dimensional data
    '''
    return np.array([[np.cos(x), np.sin(x)]]).T  # 2x1


def S1(data):
    return np.concatenate([S1_j(x) for x in data])  # len(data) x 1


def S2_jk(x, j, k):
    j -= 1
    k -= 1
    return np.array([[np.cos(x[j]-x[k]), np.sin(x[j]-x[k]), np.cos(x[j]+x[k]), np.sin(x[j]+x[k])]]).T


def S2(data):
    d = len(data)
    l = [i for i in range(1, d+1)]
    arrays = []
    for v in itertools.combinations(l, 2):
        arrays.append(S2_jk(data, v[0], v[1]))
    # print(np.concatenate(arrays).shape)
    return np.concatenate(arrays)


def S(data):  # S1は2d=6, S2は2d^2-2d=12
    '''data : 長さNの配列'''
    return np.concatenate([S1(data), S2(data)])


def H(data):
    return np.concatenate([S1(data), 2*S2(data)])


def D(x):
    '''
    Input:
        長さdの配列
    Return:
        m x d
    '''
    d = len(x)
    entries = []
    for ind in range(0, d):
        tmp_ = [0 for _ in range(d)]
        tmp_[ind] = -np.sin(x[ind])
        entries.append(tmp_)
        tmp_ = [0 for _ in range(d)]
        tmp_[ind] = np.cos(x[ind])
        entries.append(tmp_)

    l = [i for i in range(0, d)]

    for v in itertools.combinations(l, 2):
        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = -np.sin(x[v[0]]-x[v[1]])
        tmp_[v[1]] = np.sin(x[v[0]]-x[v[1]])
        entries.append(tmp_)

        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = np.cos(x[v[0]]-x[v[1]])
        tmp_[v[1]] = -np.cos(x[v[0]]-x[v[1]])
        entries.append(tmp_)

        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = -np.sin(x[v[0]]+x[v[1]])
        tmp_[v[1]] = -np.sin(x[v[0]]+x[v[1]])
        entries.append(tmp_)

        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = np.cos(x[v[0]]+x[v[1]])
        tmp_[v[1]] = np.cos(x[v[0]]+x[v[1]])
        entries.append(tmp_)
    mat = np.array(entries)
    return mat


def Gamma(x):
    '''
    Input:
        長さdの配列
    Return:
        m x m
    '''
    return D(x)@D(x).T

def core_of_torus_graph(x, phi):
    '''
    x : (2d^2,1)のnumpy配列, data
    phi : (2d^2,1)のnumpy配列

    Return
        (1,1)のnumpy配列
    '''
    x = list(x.T[0])
    core = phi.T@S(x)

    return np.exp(core)


def sample_from_torus_graph(num_samples, d, phi, verbose=True):
    '''
    num_samples : サンプリングしたい数
    d : dimension
    phi : モデルパラメタ

    '''

    assert len(phi) == 2*d*d

    # rejection samplingを行う
    def q(x):
        return 1

    phi_ = phi.flatten().tolist()  # リストに変換
    core = 0
    for ind in range(d*d):
        core += (phi_[2*ind]**2 + phi_[2*ind+1]**2)**0.5

    k = np.exp(core)  # 上界
    if verbose:
        print("upper bound constant", k)
    samples = []
    trial = 0
    acceptance = 0
    reject = 0

    while acceptance < num_samples:
        trial += 1
        # 提案分布からサンプリング. この例では一様.
        x = np.random.random_sample((d, 1)) * 2 * np.pi
        p = core_of_torus_graph(x, phi)
        u = np.random.random_sample()
        if u <= p/(k*q(x)):  # accept
            samples.append(x)
            acceptance += 1
            if verbose:
                print(f"{len(samples)}/{num_samples}")
        else:  # reject
            reject += 1
    if verbose:
        print("acceptance rate:", acceptance/trial)
    return np.concatenate(samples,axis=1).T, acceptance/trial


def torus_graph_density(phi, x1, x2):
    kernel = 0
    kernel += phi[0]*np.cos(x1)+phi[1]*np.sin(x1)
    kernel += phi[2]*np.cos(x2)+phi[3]*np.sin(x2)
    kernel += phi[4]*np.cos(x1-x2)+phi[5]*np.sin(x1-x2) + \
        phi[6]*np.cos(x1+x2)+phi[7]*np.sin(x1+x2)
    return np.exp(kernel)

#true_phi = np.array([[1,0,1,0,0,0,0,0]]).T
true_phi = np.array([[1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0]]).T

losses = []
times = []
from tqdm import tqdm
LAMBDA = 1
for _ in tqdm(range(1)):
    data_arr,acc = sample_from_torus_graph(1000,3,true_phi,False)
    start_time = time()
    est = estimate_phi_lasso(data_arr,l=LAMBDA)
    end_time = time()
    print(est.shape,true_phi.shape)
    times.append(end_time-start_time)
    losses.append(np.linalg.norm(true_phi[10:14,:] - est[8:12,:]).item()) #RMSE
print("RMSE:", sum(losses)/len(losses))
#print("total time:", end_time-start_time, " seconds")
print("per time:", sum(times)/len(times), " seconds")
print(true_phi)
print(est)




#plt.hist(losses)
#plt.savefig("rmse_dist.png")


#pdb.set_trace()
