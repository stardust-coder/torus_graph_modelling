import cupy as cp
import numpy as np
import itertools
from tqdm import tqdm
import pdb
from math import sin, cos
from time import time

def S1_j(x):
    '''
    x : d  dimensional data
    '''
    return cp.array([[cp.cos(x), cp.sin(x)]]).T  # 2x1


def S1(data):
    return cp.concatenate([S1_j(x) for x in data])  # len(data) x 1


def S2_jk(x, j, k):
    j -= 1
    k -= 1
    return cp.array([[cp.cos(x[j]-x[k]), cp.sin(x[j]-x[k]), cp.cos(x[j]+x[k]), cp.sin(x[j]+x[k])]]).T


def S2(data):
    d = len(data)
    l = [i for i in range(1, d+1)]
    arrays = []
    for v in itertools.combinations(l, 2):
        arrays.append(S2_jk(data, v[0], v[1]))
    # print(cp.concatenate(arrays).shape)
    return cp.concatenate(arrays)


def S(data):  # S1は2d=6, S2は2d^2-2d=12
    '''data : 長さNの配列'''
    return cp.concatenate([S1(data), S2(data)])


def H(data):
    return cp.concatenate([S1(data), 2*S2(data)])


def D(x):
    '''
    Icput:
        長さdの配列
    Return:
        m x d
    '''
    d = len(x)
    entries = []
    for ind in range(0, d):
        tmp_ = [0 for _ in range(d)]
        tmp_[ind] = -sin(x[ind])
        entries.append(tmp_)
        tmp_ = [0 for _ in range(d)]
        tmp_[ind] = cos(x[ind])
        entries.append(tmp_)

    l = [i for i in range(0, d)]

    for v in itertools.combinations(l, 2):
        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = -sin(x[v[0]]-x[v[1]])
        tmp_[v[1]] = sin(x[v[0]]-x[v[1]])
        entries.append(tmp_)

        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = cos(x[v[0]]-x[v[1]])
        tmp_[v[1]] = -cos(x[v[0]]-x[v[1]])
        entries.append(tmp_)

        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = -sin(x[v[0]]+x[v[1]])
        tmp_[v[1]] = -sin(x[v[0]]+x[v[1]])
        entries.append(tmp_)

        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = cos(x[v[0]]+x[v[1]])
        tmp_[v[1]] = cos(x[v[0]]+x[v[1]])
        entries.append(tmp_)
    
    mat = cp.asarray(entries)
    return mat


def Gamma(x):
    '''
    Icput:
        長さdの配列
    Return:
        m x m
    '''
    return cp.dot(D(x),D(x).T)


def estimate_phi(data):
    '''data : n x d'''
    n, d = data.shape
    Gamma_hat = cp.zeros((2*d*d, 2*d*d))
    H_hat = cp.zeros((2*d*d, 1))
    V_zero_hat = cp.zeros((2*d*d, 2*d*d))

    for ind in tqdm(range(n), desc='Estimating Phi', leave=False):
        print(time)
        x = data[ind]
        print(time())
        x = cp.asarray(x)
        print(time())
        Gamma_hat = Gamma_hat + Gamma(x)
        print(time())
        tmp_ = H(x)
        print(time())
        H_hat = H_hat + tmp_
        print(time())
        V_zero_hat = V_zero_hat + cp.dot(tmp_,tmp_.T)
        print(time())

    Gamma_hat = Gamma_hat/n
    H_hat = H_hat/n
    V_zero_hat = V_zero_hat/n
    Gamma_hat_inv = cp.linalg.inv(Gamma_hat)
    return cp.dot(Gamma_hat_inv,H_hat), Gamma_hat, Gamma_hat_inv, H_hat, V_zero_hat


def test_for_one_edge(N, d, est, a, b, sigma, verbose=False):
    '''
    Test whether two nodes, node a (1~d) and node b (1~d, b≠a), are independent.

    est: estimated value of each parameters
    N : num of samples
    '''
    assert a < b
    assert b <= d
    assert 2*d*d == len(est)

    l = [i for i in range(1, d+1)]
    ind_list = [v for v in itertools.combinations(l, 2)]
    ind_ = ind_list.index((a, b))
    phi_hat = est[2*d+4*ind_:2*d+4*(ind_+1)]
    phi_test = phi_hat.T@cp.linalg.inv(
        sigma[2*d+4*ind_:2*d+4*(ind_+1), 2*d+4*ind_:2*d+4*(ind_+1)])@phi_hat * N #calculate test statistics
    
    # visualize test statistics
    if verbose:
        print("INDEX:", ind_)
        print(ind_,2*d+4*ind_,2*d+4*(ind_+1))

        print(phi_hat.T)
        print(sigma[2*d+4*ind_:2*d+4*(ind_+1), 2*d+4*ind_:2*d+4*(ind_+1)])

        print("test statistics following chi2 distribution = ", phi_test.item())
        se = 0
        print("1-alpha percent confidential interval", phi_test.item()-1.96*se, "~",  phi_test.item()+1.96*se)
    return phi_test.item()
