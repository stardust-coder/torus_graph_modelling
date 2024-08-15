import traceback
import networkx as nx
import numpy as np
import scipy
import itertools
import matplotlib.pyplot as plt
from time import time
from parfor import parfor

def S1_j(x): #x : d  dimensional data
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


def S(data): #data : list of len(N)
    return np.concatenate([S1(data), S2(data)])


def H(data):
    return np.concatenate([S1(data), 2 * S2(data)])


def D(x): #x : list of len(d), return m x d array
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


def Gamma(x): #x : list of len(d), return mxm array
    return D(x) @ D(x).T

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


class Torus_Graph:
    def __init__(self, dim):
        self.d = dim
        d = self.d
        self.param = np.zeros((2*dim*dim,1))
        self.naive_est = np.zeros((2*dim*dim,1))
        self.naive_est_flag = False

        self.G = nx.Graph()
        self.G.add_nodes_from([i+1 for i in range(dim)])
        self.thresh = 1e-4
        self.index_dictionary = {}
        for i, v in enumerate(list(itertools.combinations(range(1, self.d + 1), 2))):
            self.index_dictionary[v] = i+1

    def edge_index(self,a,b): # a,b: integer from 1 to d
        return self.index_dictionary[(a,b)]

    def assign_by_edge(self,edge,val):
        self.param[2*self.d+4*(self.edge_index(*edge)-1):2*self.d+4*(self.edge_index(*edge)),:] = val

    def get_param_of_vec(self,t,vec):
        '''
        When len(t) == 1, get parameters corresponding to node t.
        When len(t) == 2, get parameters corresponding to node t[0] and node t[1].
        '''
        assert len(t) in [1,2]
        if len(t) == 1:
            return vec[2*(t[0]-1):2*(t[0]),:]
        elif len(t) == 2:
            return vec[2*self.d+4*(self.edge_index(t[0],t[1])-1):2*self.d+4*(self.edge_index(t[0],t[1])),:]
    
    def get_param(self,t):
        return self.get_param_of_vec(t,self.param)

    def lasso(self,data,l):
        return 
    
    def glasso(self,data,l):
        return 

    def estimate_by_edge(self, data, mode="naive"): #TODO: implement conditional distribution based parallel estimation
        return 
            
    def estimate(self, data, mode="naive"):
        assert data.shape[1] == self.d, "Data shape mismatch!"
        n, d = data.shape
        start_estimation = time()

        def calc_matrices(data): #TODO: acceleration with GPU            
            Gamma_hat = np.zeros((2 * d * d, 2 * d * d))
            H_hat = np.zeros((2 * d * d, 1))
            V_zero_hat = np.zeros((2 * d * d, 2 * d * d))
            for j in range(n):
                x = data[j]
                Gamma_hat = Gamma_hat + Gamma(x)
                tmp_ = H(x)
                H_hat = H_hat + tmp_
                V_zero_hat = V_zero_hat + tmp_ @ tmp_.T
            Gamma_hat = Gamma_hat / n
            H_hat = H_hat / n
            V_zero_hat = V_zero_hat / n
            return Gamma_hat, H_hat
    
        Gamma_hat, H_hat = calc_matrices(data)
        if mode=="naive":
            self.param = np.linalg.inv(Gamma_hat)@H_hat
            self.naive_est = self.param.copy()
            self.naive_est_flag = True
            print(self.param)

        elif mode=="lasso":
            d = self.d
            assert self.naive_est_flag != False
            
            x_admm = np.zeros((2*d*d, 1))  # warm start
            z_admm = np.zeros((2*d*d, 1))
            u_admm = np.zeros((2*d*d, 1))
            k = 0
            mu = 1  # hyperparameter
            INV = np.linalg.inv(mu * np.identity(2*d*d) + Gamma_hat)
            est_list = []
            edge_list = []
            binarr_list = []
            lambda_list = np.logspace(-2, 1, num=30).tolist()
            for l in lambda_list:
                k += 1
                iter_ = 10000# 1ならapproxiamte, 大きく設定したほうがexactに近い
                for _ in range(iter_):
                    x_new = INV @ (mu * (z_admm - u_admm) + H_hat)
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
                E = []
                B = np.ones((2*d*d,1))
                for e in list(itertools.combinations(range(1, self.d + 1), 2)):
                    if np.linalg.norm(self.get_param_of_vec((e[0],e[1]),est_with_admm_onestep)) > self.thresh:
                        E.append(e)
                    else:
                        ind = self.index_dictionary[e]
                        B[2*d+4*(ind-1):2*d+4*ind] = 0
                edge_list.append(E)
                binarr_list.append(B)
            
            # @parfor(range(len(lambda_list))) #use parfor
            def calc_SMIC(j):
                est_arr = self.naive_est * binarr_list[j]
                I = np.zeros((2*(d**2),2*(d**2)))
                Gamma_hat = np.zeros((2*(d**2),2*(d**2)))
                H_hat = np.zeros((2*(d**2), 1))
                for data_ind in range(n):
                    x = data[data_ind]
                    G_ = Gamma(x)
                    Gamma_hat = Gamma_hat + G_
                    H_ = H(x)
                    H_hat = H_hat + H_
                    tmp = G_ @ est_arr - H_
                    I = I + tmp @ tmp.T
                I = I / n
                Gamma_hat = Gamma_hat/n
                H_hat = H_hat/n
                smic1 = n*(-est_arr.T@H_hat)  #plugged-in optimal estimator to quaratic form
                smic1 = smic1.item()
                eigvals = scipy.linalg.eigh(I,Gamma_hat,eigvals_only=True)
                smic2 = sum(eigvals)
                smic = smic1 + smic2
                return smic

            scores = [calc_SMIC(j) for j in range(len(lambda_list))]
            # scores = calc_SMIC #use parfor

            opt_index = scores.index(min(scores))
            self.param = est_list[opt_index]

            ### print estimated results
            r_prev = (None,None,None,None)
            for i in range(len(lambda_list)):
                r_new = lambda_list[i],scores[i], edge_list[i],est_list[i].T
                if edge_list[i] == edge_list[opt_index]:
                    print("[OPTIMAL GRAPH STRUCTURE with the smallest SMIC]")
                if r_new[2] == r_prev[2]:
                    pass
                else:
                    print(r_new)
                    r_prev = r_new

        # elif mode=="glasso":

        end_estimation = time()
        for e in list(itertools.combinations(range(1, self.d + 1), 2)):
            weight = np.linalg.norm(self.get_param((e[0],e[1])))
            if weight > self.thresh:
                self.G.add_edge(e[0],e[1],weight=weight)
            else:
                try:
                    self.G.remove_edge(e[0],e[1])
                except:
                    pass

        print("Estimation time(s):",end_estimation-start_estimation)
        
    def graph_property(self):
        print("Average clustering coefficient = ", nx.average_clustering(self.G))
        print("Average shortest path length = ", nx.average_shortest_path_length(self.G))
        print("Small-world index = ", nx.sigma(self.G),nx.omega(self.G))

    def plot(self, weight=False):
        weights = nx.get_edge_attributes(self.G, 'weight').values()
        pos = nx.circular_layout(self.G)
        plt.figure(figsize=(10,10)) #グラフエリアのサイズ
        cmap=plt.cm.RdBu_r
        if weight:
            nx.draw_networkx(self.G, pos,node_color = 'w', edge_color = weights, edge_cmap = cmap)
        else:
            nx.draw_networkx(self.G, pos,node_color = 'w')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm,ax=plt.gca())
        plt.show()
        plt.savefig("graph.png")
