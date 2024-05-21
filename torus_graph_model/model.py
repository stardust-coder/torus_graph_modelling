import cupy as cp
import numpy as np
import itertools
from tqdm import tqdm
from time import time
import pdb

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


#lasso with cordinate descent

def estimate_phi_lasso(data, l):
    n, d = data.shape
    Gamma_hat = np.zeros((2*d*d, 2*d*d))
    H_hat = np.zeros((2*d*d, 1))
    V_zero_hat = np.zeros((2*d*d, 2*d*d))

    ###時間かかる. なぜ？
    for ind in tqdm(range(n), desc='Estimating Phi', leave=False):
        x = data[ind]
        Gamma_hat = Gamma_hat + Gamma(x)
        tmp_ = H(x)
        H_hat = H_hat + tmp_
        V_zero_hat = V_zero_hat + tmp_@tmp_.T
    Gamma_hat = Gamma_hat/n
    H_hat = H_hat/n
    V_zero_hat = V_zero_hat/n

    #phi_tmp = estimate_phi(data)[0] #computationally costly
    
    phi_tmp = np.zeros((2*d*d,1))
    LAMBDA = l
    for _ in range(1000):
        print(phi_tmp)
        for j in range(2*d*d):
            if phi_tmp[j][0] > 0:
                phi_tmp[j][0] = (-(Gamma_hat[j,:]@phi_tmp - Gamma_hat[j][j]*phi_tmp[j][0].item()) + H_hat[j][0] - LAMBDA)/Gamma_hat[j][j]
            else:
                phi_tmp[j][0] = (-(Gamma_hat[j,:]@phi_tmp - Gamma_hat[j][j]*phi_tmp[j][0].item()) + H_hat[j][0] + LAMBDA)/Gamma_hat[j][j]        
        
    return phi_tmp


def estimate_phi(data):
    '''data : n x d'''
    n, d = data.shape

    #pattern1
    Gamma_hat = np.zeros((2*d*d, 2*d*d))
    H_hat = np.zeros((2*d*d, 1))
    V_zero_hat = np.zeros((2*d*d, 2*d*d))

    for ind in tqdm(range(n), desc='Estimating Phi', leave=False):
        x = data[ind]
        Gamma_hat = Gamma_hat + Gamma(x)
        tmp_ = H(x)
        H_hat = H_hat + tmp_
        V_zero_hat = V_zero_hat + tmp_@tmp_.T
    Gamma_hat = Gamma_hat/n
    H_hat = H_hat/n
    V_zero_hat = V_zero_hat/n
    Gamma_hat_inv = np.linalg.inv(Gamma_hat)

    return Gamma_hat_inv@H_hat, Gamma_hat, Gamma_hat_inv, H_hat, V_zero_hat

    #pattern2
    # A = []
    # Gammas = []
    # for ind in tqdm(range(n), desc='Estimating Phi', leave=False):
    #     x = data[ind]
    #     Gammas.append(Gamma(x))
    #     tmp_ = H(x)
    #     A.append(tmp_.T.flatten().tolist())
    # A = cp.array(A)
    # V_zero_hat_gpu = cp.dot(A.T,A)
    # V_zero_hat = cp.asnumpy(V_zero_hat_gpu)
    # print("A)",time())
    # Gamma_hat = np.sum(Gammas)
    # print("B)",time())
    # Gamma_hat = Gamma_hat/n
    # H_hat = H_hat/n
    # V_zero_hat = V_zero_hat/n
    # Gamma_hat_inv = np.linalg.inv(Gamma_hat)
    # print("C)",time())
    # return Gamma_hat_inv@H_hat, Gamma_hat, Gamma_hat_inv, H_hat, V_zero_hat



def estimate_phi_numpy(data):
    '''data : n x d'''
    n, d = data.shape
    def Gamma_(x):
        return np.outer(x, x)

    def estimate_parameters(data):
        print(time())
        Gamma_x = [Gamma_(x) for x in data]  # 各xに対するGamma(x)を計算し、配列として保持
        print(time())
        #Gamma_hat = np.sum(np.kron(gx, gx) for gx in Gamma_x)
        Gamma_hat = cp.sum([cp.kron(cp.asarray(gx), cp.asarray(gx)) for gx in Gamma_x])
        Gamma_hat = cp.asnumpy(Gamma_hat)
        print(time())
        H_hat = np.sum(gx.flatten()[:, np.newaxis] for gx in Gamma_x)
        print(time())
        V_zero_hat = np.sum(np.outer(gx.flatten(), gx.flatten()) for gx in Gamma_x)
        print(time())

        return Gamma_hat, H_hat, V_zero_hat
    
    Gamma_hat, H_hat, V_zero_hat = estimate_parameters(data)
    Gaa_hat = Gamma_hat/n
    H_hat = H_hat/n
    V_zero_hat = V_zero_hat/n
    Gamma_hat_inv = np.linalg.inv(Gamma_hat)
    return Gamma_hat_inv@H_hat, Gamma_hat, Gamma_hat_inv, H_hat, V_zero_hat


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
    phi_test = phi_hat.T@np.linalg.inv(
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
