import numpy as np
import itertools
from tqdm import tqdm


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


def estimate_phi(data):
    '''data : n x d'''
    n, d = data.shape
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
