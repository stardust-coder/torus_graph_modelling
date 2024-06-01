#import cupy as cp
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




#lasso with conditional distribution
def estimate_phi_lasso(data, l):
    n,d = data.shape

    node_to_vec = [[] for _ in range(d)]
    for i,v in enumerate(itertools.combinations(range(1,d+1), 2)):
        node_to_vec[v[0]-1].append(i)
        node_to_vec[v[1]-1].append(i)

    def get_indices(a,b): #a,b: from 1 to d
        #ind_list = [v for v in itertools.combinations(range(1,d+1), 2)]
        #remove_ind = ind_list.index((a, b))        
        indices = []
        indices.append(2*(a-1))
        indices.append(2*(a-1)+1)
        indices.append(2*(b-1))
        indices.append(2*(b-1)+1)
        for item in node_to_vec[a-1]:
            #if item != remove_ind:
            indices.extend(range(2*d + 4*item,2*d + 4*(item+1) ))
        for item in node_to_vec[b-1]:
            #if item != remove_ind:
            indices.extend(range(2*d + 4*item,2*d + 4*(item+1) ))
        indices = sorted(list(set(indices)))
        return indices
    
    def est_partial(a,b):
        index = get_indices(a,b)
        Gamma_hat = np.zeros((2*d*d, 2*d*d))
        Gamma_hat_small = np.zeros((8*(d-1),8*(d-1)))
        H_hat = np.zeros((2*d*d, 1))
        #V_zero_hat = np.zeros((2*d*d, 2*d*d))
        for ind in tqdm(range(n), desc='Estimating Phi', leave=False):
            #full
            x = data[ind]
            Gamma_hat = Gamma_hat + Gamma(x)
            
            #conditional ver1
            x = data[ind]
            D_small = D(x)[np.ix_(index,[a-1,b-1])]        
            Gamma_hat_small =  Gamma_hat_small + D_small@D_small.T
            
            tmp_ = H(x)
            H_hat = H_hat + tmp_
            #V_zero_hat = V_zero_hat + tmp_@tmp_.T
        Gamma_hat = Gamma_hat/n
        Gamma_hat_small = Gamma_hat_small/n
        H_hat = H_hat/n
        
        #V_zero_hat = V_zero_hat/n    
    
        #正しい推定値。計算が重い。
        #print(np.linalg.inv(Gamma_hat)@H_hat) #全体
        #est_without_lasso = np.linalg.inv(Gamma_hat[np.ix_(index,index)])@H_hat[index]  #正しい
        #est_without_lasso = np.linalg.inv(Gamma_hat_small)@H_hat[index]#あやまり
        #print(est_without_lasso) #正しい


        #print("phiとノード番号との対応関係")
        #print(index)
        #print([a,a,b,b])
        #print([list(itertools.combinations(range(1,d+1), 2))[int((x-2*d)/4)] for x in index[4:]])
    

        #ADMMでlassoを実行
        def soft_threshold(param,t_vec):
            res = np.zeros(t_vec.shape)
            for i,t in enumerate(t_vec.flatten().tolist()):
                if t > param:
                    res[i][0] = t-param
                elif t < -param:
                    res[i][0] =  t+param
                else:
                    res[i][0] = 0
            return res
        
        d_ = 8*(d-1)
        x_admm = np.zeros((d_,1)) #not used
        z_admm = np.zeros((d_,1))
        u_admm = np.zeros((d_,1))
        k = 0
        mu = 1 #hyperparameter
        
        print(f"Calculating {d_} times {d_} matrix inversion...")
        print(time())
        INV = np.linalg.inv(mu*np.identity(d_) + Gamma_hat[np.ix_(index,index)])
        print(time())

        while True:
            k += 1
            x_new = INV@(mu*(z_admm-u_admm)+H_hat[index])
            x_dif = np.linalg.norm(x_new-x_admm)
            z_new = soft_threshold(l/mu,x_new+u_admm)
            z_dif = np.linalg.norm(z_new-z_admm)
            r_dif = np.linalg.norm(x_new-z_new)
            u_new = u_admm + x_new - z_new
            x_admm = x_new.copy()
            z_admm = z_new.copy()
            u_admm = u_new.copy()
            if z_dif < 1e-7 and x_dif < 1e-7 and np.linalg.norm(z_admm-x_admm) < 1e-7:
               break
            if k >= 10000:
                break
            print("iter=",k,"x_dif=",x_dif,"z_dif=",z_dif,"residual=",r_dif)
            # if r_dif > 10 * mu*z_dif:
            #     mu = mu*2
            #     print("mu increased to ",mu)
            # elif mu*z_dif > 10 * r_dif:
            #     mu = mu/2
            #     print("mu decreased to ",mu)
            # else:
            #     mu = mu
            
            # print("Objective:", x_new.T@Gamma_hat_small@x_new/2 - x_new.T@H_hat[index] + l * np.linalg.norm(x_new,ord=1))
            # print("Objective:", x_new.T@Gamma_hat_small@x_new/2 - x_new.T@H_hat[index])
            # true_phi = np.array([[1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0]]).T[index]
            # print("True:", true_phi.T@Gamma_hat_small@true_phi/2 - true_phi.T@H_hat[index] + l * np.linalg.norm(true_phi,ord=1))
            # print("True:", true_phi.T@Gamma_hat_small@true_phi/2 - true_phi.T@H_hat[index])
            
        est_with_lasso = x_admm.copy()
        # import pdb; pdb.set_trace()


        # #gradient descentでlassoを実行
        # current_phi = np.zeros((8*(d-1),1))
        # k = 0
        # while True:
        #     k += 1
        #     #print(k)
        #     grad = Gamma_hat_small @ current_phi - H_hat[index] + np.sign(current_phi) * l
        #     alpha = 1e-2
        #     diff = alpha * grad
        #     current_phi = current_phi - alpha * grad
        #     #print(current_phi)
        #     #print(np.linalg.norm(diff))
        #     if np.linalg.norm(diff) < 1e-2 or k > 10000:
        #         print(f"Terminated in {k} steps.")
        #         break
        #     #目的関数は着実に減少していっている.
        #     print(current_phi.T@Gamma_hat_small@current_phi/2 - current_phi.T@H_hat[index] + l * np.linalg.norm(current_phi,ord=1))
        #     # pdb.set_trace()
        # est_with_lasso = current_phi

        # #import pdb; pdb.set_trace()
        # #print(est_with_lasso)
        
        return est_with_lasso
    
    # return est_partial(1,3) #simulation評価用
    
    res = {}
    ind_list = list(itertools.combinations(range(1,d+1), 2))
    for i,t in enumerate(ind_list):
       tmp = [x for x in ind_list if t[0] in list(x) or t[1] in list(x) ]
       tmp.sort()
       ind = tmp.index(t)
       res[t] = est_partial(t[0],t[1])[2*2+4*(ind):2*2+4*(ind+1)]
    return res #dictで返す. (node a, node b)のkeyで推定値がvalue.

#lasso with cordinate descent

def estimate_phi_lasso2(data, l):
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
