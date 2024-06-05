# import cupy as cp
import numpy as np
import itertools
from tqdm import tqdm
from time import time
import pdb


def S1_j(x):
    """
    x : d  dimensional data
    """
    return np.array([[np.cos(x), np.sin(x)]]).T  # 2x1


def S1(data):
    return np.concatenate([S1_j(x) for x in data])  # len(data) x 1


def S2_jk(x, j, k):
    j -= 1
    k -= 1
    return np.array(
        [
            [
                np.cos(x[j] - x[k]),
                np.sin(x[j] - x[k]),
                np.cos(x[j] + x[k]),
                np.sin(x[j] + x[k]),
            ]
        ]
    ).T


def S2(data):
    d = len(data)
    l = [i for i in range(1, d + 1)]
    arrays = []
    for v in itertools.combinations(l, 2):
        arrays.append(S2_jk(data, v[0], v[1]))
    # print(np.concatenate(arrays).shape)
    return np.concatenate(arrays)


def S(data):  # S1は2d=6, S2は2d^2-2d=12
    """data : 長さNの配列"""
    return np.concatenate([S1(data), S2(data)])


def H(data):
    return np.concatenate([S1(data), 2 * S2(data)])


def D(x):
    """
    Input:
        長さdの配列
    Return:
        m x d
    """
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
        tmp_[v[0]] = -np.sin(x[v[0]] - x[v[1]])
        tmp_[v[1]] = np.sin(x[v[0]] - x[v[1]])
        entries.append(tmp_)

        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = np.cos(x[v[0]] - x[v[1]])
        tmp_[v[1]] = -np.cos(x[v[0]] - x[v[1]])
        entries.append(tmp_)

        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = -np.sin(x[v[0]] + x[v[1]])
        tmp_[v[1]] = -np.sin(x[v[0]] + x[v[1]])
        entries.append(tmp_)

        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = np.cos(x[v[0]] + x[v[1]])
        tmp_[v[1]] = np.cos(x[v[0]] + x[v[1]])
        entries.append(tmp_)
    mat = np.array(entries)
    return mat


def Gamma(x):
    """
    Input:
        長さdの配列
    Return:
        m x m
    """
    return D(x) @ D(x).T


#naive matrix inversion estimation
def estimate_phi_naive(data):
    n, d = data.shape
    Gamma_hat = np.zeros((2 * d * d, 2 * d * d))
    H_hat = np.zeros((2 * d * d, 1))
    V_zero_hat = np.zeros((2 * d * d, 2 * d * d))
    for j in tqdm(range(n), desc="Estimating Phi", leave=False):
        x = data[j]
        Gamma_hat = Gamma_hat + Gamma(x)
        tmp_ = H(x)
        H_hat = H_hat + tmp_
        V_zero_hat = V_zero_hat + tmp_ @ tmp_.T
    Gamma_hat = Gamma_hat / n
    H_hat = H_hat / n
    V_zero_hat = V_zero_hat / n
    print(f"Inversion of {2*d*d} x {2*d*d} matrix...")
    Gamma_hat_inv = np.linalg.inv(Gamma_hat)
    print("End.")
    res_mat = Gamma_hat_inv @ H_hat
    #return res_mat

    res_dict = {}
    ind_list = list(itertools.combinations(range(1, d + 1), 2))
    for i,t in enumerate(range(1,d+1)):
        res_dict[(0,t)] = res_mat[2*i:2*(i+1)]
    for i, t in tqdm(enumerate(ind_list)):
        res_dict[(t[0], t[1])] = res_mat[2 * d + 4 * (i) : 2 * d + 4 * (i + 1)]
    return res_dict

def get_indices(d,a,b):  # a,b: from 1 to d
    node_to_vec = [[] for _ in range(d)]
    for i, v in enumerate(itertools.combinations(range(1, d + 1), 2)):
        node_to_vec[v[0] - 1].append(i)
        node_to_vec[v[1] - 1].append(i)
    indices = []
    indices.append(2 * (a - 1))
    indices.append(2 * (a - 1) + 1)
    indices.append(2 * (b - 1))
    indices.append(2 * (b - 1) + 1)
    for item in node_to_vec[a - 1]:
        indices.extend(range(2 * d + 4 * item, 2 * d + 4 * (item + 1)))
    for item in node_to_vec[b - 1]:
        indices.extend(range(2 * d + 4 * item, 2 * d + 4 * (item + 1)))
    indices = sorted(list(set(indices)))
    return indices

def dict_to_mat(d):
    return np.array([[0]])

# using conditional distribution without lasso
def estimate_phi(data,invalid_edges=[]):
    n, d = data.shape

    def est_partial(a, b):
        index = get_indices(d,a,b)
        # Gamma_hat = np.zeros((2 * d * d, 2 * d * d))
        Gamma_hat_small = np.zeros((8 * (d - 1), 8 * (d - 1)))
        H_hat = np.zeros((2 * d * d, 1))
        for j in tqdm(range(n), desc="Estimating Phi", leave=False):
            x = data[j]
            D_small = D(x)[index, :]
            # Gamma_hat = Gamma_hat + Gamma(x)
            Gamma_hat_small = Gamma_hat_small + D_small @ D_small.T
            tmp_ = H(x)
            H_hat = H_hat + tmp_
        # Gamma_hat = Gamma_hat / n
        Gamma_hat_small = Gamma_hat_small / n
        H_hat = H_hat / n
        est_without_lasso = np.linalg.inv(Gamma_hat_small) @ H_hat[index]
        return est_without_lasso
    # return est_partial(1,3) #simulation評価用

    res_dict = {}
    ind_list = list(itertools.combinations(range(1, d + 1), 2))
    for i, t in enumerate(ind_list):
        res_dict[0, t[0]] = np.zeros((2,1))
        res_dict[0, t[1]] = np.zeros((2,1))
        res_dict[t] = np.zeros((4,1))
        if t not in invalid_edges:
            tmp = [x for x in ind_list if t[0] in list(x) or t[1] in list(x)]
            tmp.sort()
            ind = tmp.index(t)
            est_p = est_partial(t[0], t[1])
            res_dict[0, t[0]] = est_p[:2]
            res_dict[0, t[1]] = est_p[2:4]
            res_dict[t] = est_p[2 * 2 + 4 * (ind) : 2 * 2 + 4 * (ind + 1)]
    return res_dict  # dictで返す. (node a, node b)のkeyで推定値がvalue.


# lasso with ADMM
def estimate_phi_admm(data, l):
    n, d = data.shape

    def est_partial(a, b):
        index = get_indices(d, a, b)
        Gamma_hat_small = np.zeros((8 * (d - 1), 8 * (d - 1)))
        H_hat = np.zeros((2 * d * d, 1))
        for ind in tqdm(range(n), desc="Estimating Phi", leave=False):
            x = data[ind]
            D_small = D(x)[index, :]
            Gamma_hat_small = Gamma_hat_small + D_small @ D_small.T

            tmp_ = H(x)
            H_hat = H_hat + tmp_
        Gamma_hat_small = Gamma_hat_small / n
        H_hat = H_hat / n

        # ADMMでlassoを実行
        def soft_threshold(param, t_vec):
            res = np.zeros(t_vec.shape)
            for i, t in enumerate(t_vec.flatten().tolist()):
                if t > param:
                    res[i][0] = t - param
                elif t < -param:
                    res[i][0] = t + param
                else:
                    res[i][0] = 0
            return res

        d_ = 8 * (d - 1)
        x_admm = np.zeros((d_, 1)) 
        z_admm = np.zeros((d_, 1))
        u_admm = np.zeros((d_, 1))
        k = 0
        mu = 1  # hyperparameter

        print(f"Calculating {d_} times {d_} matrix inversion...")
        INV = np.linalg.inv(mu * np.identity(d_) + Gamma_hat_small)
        print("End.")

        while True:
            k += 1
            x_new = INV @ (mu * (z_admm - u_admm) + H_hat[index])
            x_dif = np.linalg.norm(x_new - x_admm)
            z_new = soft_threshold(l / mu, x_new + u_admm)
            z_dif = np.linalg.norm(z_new - z_admm)
            r_dif = np.linalg.norm(x_new - z_new)
            u_new = u_admm + x_new - z_new
            x_admm = x_new.copy()
            z_admm = z_new.copy()
            u_admm = u_new.copy()
            if z_dif < 1e-7 and x_dif < 1e-7 and np.linalg.norm(z_admm - x_admm) < 1e-7:
                break
            if k >= 100000:
                break
            # print("iter=", k, "x_dif=", x_dif, "z_dif=", z_dif, "residual=", r_dif)
        est_with_lasso = z_admm.copy()
        return est_with_lasso

    # return est_partial(1,3) #simulation評価用
    res_dict = {}
    ind_list = list(itertools.combinations(range(1, d + 1), 2))
    for i, t in enumerate(ind_list):
        tmp = [x for x in ind_list if t[0] in list(x) or t[1] in list(x)]
        tmp.sort()
        ind = tmp.index(t)
        est_p = est_partial(t[0], t[1])
        res_dict[0, t[0]] = est_p[:2]
        res_dict[0, t[1]] = est_p[2:4]
        res_dict[t] = est_p[2 * 2 + 4 * (ind) : 2 * 2 + 4 * (ind + 1)]
    return res_dict  # dictで返す. (node a, node b)のkeyで推定値がvalue.


def estimate_phi_admm_path(data):
    n, d = data.shape
    lambda_list = np.logspace(-2, 1, num=30).tolist()
    
    def est_partial(a, b):
        index = get_indices(d, a, b)
        Gamma_hat_small = np.zeros((8 * (d - 1), 8 * (d - 1)))
        H_hat = np.zeros((2 * d * d, 1))
        for ind in tqdm(range(n), desc="Calculating Gamma_hat_small", leave=False):
            x = data[ind]
            D_small = D(x)[index, :]
            Gamma_hat_small = Gamma_hat_small + D_small @ D_small.T
            tmp_ = H(x)
            H_hat = H_hat + tmp_
        Gamma_hat_small = Gamma_hat_small / n
        H_hat = H_hat / n

        # ADMMでlassoを実行
        def soft_threshold(param, t_vec):
            res = np.zeros(t_vec.shape)
            for i, t in enumerate(t_vec.flatten().tolist()):
                if t > param:
                    res[i][0] = t - param
                elif t < -param:
                    res[i][0] = t + param
                else:
                    res[i][0] = 0
            return res

        d_ = 8 * (d - 1)
        x_admm = np.zeros((d_, 1))  # warm start
        z_admm = np.zeros((d_, 1))
        u_admm = np.zeros((d_, 1))
        k = 0
        mu = 1  # hyperparameter
        INV = np.linalg.inv(mu * np.identity(d_) + Gamma_hat_small)

        est_list = []

        for l in lambda_list:
            k += 1
            iter_ = 1
            for _ in range(iter_):
                x_new = INV @ (mu * (z_admm - u_admm) + H_hat[index])
                x_dif = np.linalg.norm(x_new - x_admm)
                z_new = soft_threshold(l / mu, x_new + u_admm)
                z_dif = np.linalg.norm(z_new - z_admm)
                r_dif = np.linalg.norm(x_new - z_new)
                u_new = u_admm + x_new - z_new
                x_admm = x_new.copy()
                z_admm = z_new.copy()
                u_admm = u_new.copy()
                if r_dif < 1e-4 and z_dif < 1e-4:
                    break
            est_with_admm_onestep = z_admm.copy()
            est_list.append(est_with_admm_onestep)
        return est_list

    ind_list = list(itertools.combinations(range(1, d + 1), 2))
    res = [{} for _ in range(30)]
    for i, t in tqdm(enumerate(ind_list)):
        tmp = [x for x in ind_list if t[0] in list(x) or t[1] in list(x)]
        tmp.sort()
        ind = tmp.index(t)
        est_p = est_partial(t[0], t[1])
        for j in range(len(est_p)):
            res[j][0, t[0]] = est_p[j][:2]
            res[j][0, t[1]] = est_p[j][2:4]
            res[j][t[0], t[1]] = est_p[j][2 * 2 + 4 * (ind) : 2 * 2 + 4 * (ind + 1)]
    return res, lambda_list



def test_for_one_edge(N, d, est, a, b, sigma, verbose=False):
    """
    Test whether two nodes, node a (1~d) and node b (1~d, b≠a), are independent.

    est: estimated value of each parameters
    N : num of samples
    """
    assert a < b
    assert b <= d
    assert 2 * d * d == len(est)

    l = [i for i in range(1, d + 1)]
    ind_list = [v for v in itertools.combinations(l, 2)]
    ind_ = ind_list.index((a, b))
    phi_hat = est[2 * d + 4 * ind_ : 2 * d + 4 * (ind_ + 1)]
    phi_test = (
        phi_hat.T
        @ np.linalg.inv(
            sigma[
                2 * d + 4 * ind_ : 2 * d + 4 * (ind_ + 1),
                2 * d + 4 * ind_ : 2 * d + 4 * (ind_ + 1),
            ]
        )
        @ phi_hat
        * N
    )  # calculate test statistics

    # visualize test statistics
    if verbose:
        print("INDEX:", ind_)
        print(ind_, 2 * d + 4 * ind_, 2 * d + 4 * (ind_ + 1))

        print(phi_hat.T)
        print(
            sigma[
                2 * d + 4 * ind_ : 2 * d + 4 * (ind_ + 1),
                2 * d + 4 * ind_ : 2 * d + 4 * (ind_ + 1),
            ]
        )

        print("test statistics following chi2 distribution = ", phi_test.item())
        se = 0
        print(
            "1-alpha percent confidential interval",
            phi_test.item() - 1.96 * se,
            "~",
            phi_test.item() + 1.96 * se,
        )
    return phi_test.item()
