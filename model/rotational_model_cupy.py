import traceback
import networkx as nx
import numpy
import cupy as cp
import scipy
import itertools
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import os
import math
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

def S1_j(x): #x : d  dimensional data
    return cp.array([[cp.cos(x), cp.sin(x)]]).T  # 2x1


def S1(data):
    return cp.concatenate([S1_j(x) for x in data])  # len(data) x 1


def S2_jk(x, j, k):
    j -= 1
    k -= 1
    return cp.array(
        [
            [
                cp.cos(x[j] - x[k]),
                cp.sin(x[j] - x[k]),
            ]
        ]
    ).T


def S2(data):
    d = len(data)
    l = [i for i in range(1, d + 1)]
    arrays = []
    for v in itertools.combinations(l, 2):
        arrays.append(S2_jk(data, v[0], v[1]))
    return cp.concatenate(arrays)


def S(data): #data : list of len(N)
    return cp.concatenate([S1(data), S2(data)])


# def H(data):
#     return cp.concatenate([S1(data), 2 * S2(data)])


def H(data):
    def S1_j(x): #x : d  dimensional data
        return [[math.cos(x)], [math.sin(x)]]  # 2x 1
    def S1(data):
        res = []
        for x in data:
            res.extend(S1_j(x))
        return res    
    def S2_2_jk(x, j, k):
        j -= 1
        k -= 1
        return [[2*math.cos(x[j] - x[k])],[2*math.sin(x[j] - x[k])]]
    def S2_2(data):
        res = []
        for v in itertools.combinations([i for i in range(1, len(data) + 1)], 2):
            res.extend(S2_2_jk(data, v[0], v[1]))
        return res
    
    res = S1(data)
    res.extend(S2_2(data))
    res = cp.array(res)
    return res

def D(x): #x : list of len(d), return m x d array

    d = len(x)
    entries = []
    for ind in range(0, d):
        tmp_ = [0 for _ in range(d)]
        tmp_[ind] = -math.sin(x[ind])
        entries.append(tmp_)
        tmp_ = [0 for _ in range(d)]
        tmp_[ind] = math.cos(x[ind])
        entries.append(tmp_)

    for v in itertools.combinations([i for i in range(0, d)], 2):
        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = -math.sin(x[v[0]] - x[v[1]])
        tmp_[v[1]] = math.sin(x[v[0]] - x[v[1]])
        entries.append(tmp_)

        tmp_ = [0 for _ in range(d)]
        tmp_[v[0]] = math.cos(x[v[0]] - x[v[1]])
        tmp_[v[1]] = -math.cos(x[v[0]] - x[v[1]])
        entries.append(tmp_)
    
    # mat = numpy.array(entries)
    # mat_cu = cp.asarray(mat)
    mat_cu = cp.array(entries)
    return mat_cu




def Gamma(x): #x : list of len(d), return mxm array
    return D(x) @ D(x).T

def soft_threshold(param, t_vec):
    res = cp.zeros(t_vec.shape)
    for i, t in enumerate(t_vec.flatten().tolist()):
        if t > param:
            res[i][0] = t - param
        elif t < -param:
            res[i][0] = t + param
        else:
            res[i][0] = 0
    return res

def shrinkage_operator(param, x):
    coef = 1 - param/cp.linalg.norm(x).item()
    if coef < 0:
        coef = 0
    return coef * x


def model_to_indices(l):
    return [i for i, x in enumerate(l) if x != 0]
    
class Torus_Graph_Model:
    def __init__(self, dim):
        self.d = dim
        self.single_param_dim = 2
        self.pairwise_param_dim = 4
        self.initialize()

    def initialize(self):
        self.model_d = self.single_param_dim * self.d + self.pairwise_param_dim * int(self.d*(self.d-1)/2)
        self.param = cp.zeros((self.model_d,1))
        self.naive_est = cp.zeros((self.model_d,1))
        self.naive_est_flag = False

        #Calculate matrices
        self.Gamma_hat =  cp.zeros((self.model_d,1))
        self.H_hat = cp.zeros((self.model_d,1))

        self.smic = []
        self.reg_path = []
        self.est_path = []
        self.bin_path = []

        #Graph
        self.G = nx.Graph()
        self.G.add_nodes_from([i+1 for i in range(self.d)])
        self.pos = nx.circular_layout(self.G)

        #for SMIC calculations
        self.lambda_list = cp.logspace(-2, 1, num=30).tolist()
        self.glasso_weight = [1 for _ in range(self.model_d)] #weight of regularization on each group
        self.thresh = 1e-4
        self.index_dictionary = {}
        for i, v in enumerate(list(itertools.combinations(range(1, self.d + 1), 2))):
            self.index_dictionary[v] = i+1

    def edge_index(self,a,b): # a,b: integer from 1 to d
        return self.index_dictionary[(a,b)]

    def assign_by_edge(self,edge,val):
        tmp_ = self.single_param_dim*self.d+self.pairwise_param_dim*(self.edge_index(*edge)-1)
        self.param[tmp_:tmp_+self.pairwise_param_dim,:] = val

    def get_param_of_vec(self,t,vec):
        '''
        When len(t) == 1, get parameters corresponding to node t.
        When len(t) == 2, get parameters corresponding to node t[0] and node t[1].
        '''
        assert len(t) in [1,2]
        if len(t) == 1:
            return vec[self.single_param_dim*(t[0]-1):self.single_param_dim*(t[0]),:]
        elif len(t) == 2:
            tmp_ = self.single_param_dim*self.d+self.pairwise_param_dim*(self.edge_index(t[0],t[1])-1)
            return vec[tmp_:tmp_+self.pairwise_param_dim,:]
    
    def get_param(self,t):
        return self.get_param_of_vec(t,self.param)

    def estimate_by_edge(self, data, mode="naive"): #TODO: implement conditional distribution based parallel estimation
        n, d = data.shape
        start_estimation = time()

        def get_indices(d,a,b):
            node_to_vec = [[] for _ in range(d)]
            for i, v in enumerate(itertools.combinations(range(1, d + 1), 2)):
                node_to_vec[v[0] - 1].append(i)
                node_to_vec[v[1] - 1].append(i)
            indices = []
            indices.append(self.single_param_dim * (a - 1))
            indices.append(self.single_param_dim * (a - 1) + 1)
            indices.append(self.single_param_dim * (b - 1))
            indices.append(self.single_param_dim * (b - 1) + 1)
            for item in node_to_vec[a - 1]:
                indices.extend(range(self.single_param_dim * d + self.pairwise_param_dim * item, self.single_param_dim * d + self.pairwise_param_dim * (item + 1)))
            for item in node_to_vec[b - 1]:
                indices.extend(range(self.single_param_dim * d + self.pairwise_param_dim * item, self.single_param_dim * d + self.pairwise_param_dim * (item + 1)))
            indices = sorted(list(set(indices)))
            return indices

        def est_one_edge(a, b):
            d = self.d
            index = get_indices(d,a,b)
            Gamma_hat_small = cp.zeros((8 * (d - 1), 8 * (d - 1)))
            H_hat = cp.zeros((self.model_d, 1))
            for j in range(n):
                x = data[j]
                D_small = D(x)[index, :]
                Gamma_hat_small = Gamma_hat_small + D_small @ D_small.T
                tmp_ = H(x)
                H_hat = H_hat + tmp_
            Gamma_hat_small = Gamma_hat_small / n
            H_hat = H_hat / n
            est_ = cp.linalg.solve(Gamma_hat_small, H_hat[index]) #cp.linalg.inv(Gamma_hat_small) @ H_hat[index]
            self.param[index] = est_
            return est_

        ####idk which one is faster in higher dimensions
        # for i, v in enumerate(itertools.combinations(range(1, d + 1), 2)):
        #     est_one_edge(v[0],v[1])
        #     print("Estimation for:", v)
        
        tmp = Parallel(n_jobs=-1,require='sharedmem')(delayed(est_one_edge)(v[0],v[1]) for v in itertools.combinations(range(1, d + 1), 2)) #use joblib, slow because it uses shared memory

        end_estimation = time()
        self.param_to_graph()
        print("Estimation time(s):",end_estimation-start_estimation)
            
    def estimate(self, data, mode="naive", img_path="#edges_vs_SMIC.png"):
        assert data.shape[1] == self.d, "Data shape mismatch!"
        n, d = data.shape
        start_estimation = time()

        def calc_matrices(data): 
            Gamma_hat = cp.zeros((self.model_d, self.model_d))
            H_hat = cp.zeros((self.model_d, 1))
            V_zero_hat = cp.zeros((self.model_d, self.model_d)) 
            #TODO: takes time, need acceleration
            for j in tqdm(range(n)): 
                x = data[j]
                Gamma_hat = Gamma_hat + Gamma(x)
                tmp_ = H(x)
                H_hat = H_hat + tmp_
                V_zero_hat = V_zero_hat + tmp_ @ tmp_.T
            Gamma_hat /= n
            H_hat /= n
            V_hat /= n
            return Gamma_hat, H_hat

        def calc_Gamma_hat(data): 
            pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            cp.cuda.set_allocator(pool.malloc)
            D_concat = cp.empty((self.model_d,self.d*n))
            for j in tqdm(range(n)):
                D_concat[:,self.d*j:self.d*(j+1)] = D(data[j])/(n**0.5) #0.05s
            Gamma_hat = D_concat@D_concat.T
            
            del D_concat
            return Gamma_hat
        
        def calc_H_hat(data):
            H_concat = cp.empty((self.model_d,n))
            for j in tqdm(range(n)):
                H_concat[:,j:j+1]= H(data[j])/n
            H_hat = H_concat@cp.ones((n,1))
            del H_concat
            return H_hat

        if mode=="naive":
            print("Running naive estimation...")
            self.H_hat = calc_H_hat(data)
            self.Gamma_hat = calc_Gamma_hat(data)
            self.param = cp.linalg.solve(self.Gamma_hat,self.H_hat) #cp.linalg.inv(Gamma_hat)@H_hat 
            self.naive_est = self.param.copy()
            self.naive_est_flag = True
            print("\nEstimated parameters:\n")
            print(self.param.T.tolist()[0])

        elif mode=="lasso" or mode=="glasso":
            print(f"Running {mode} on full model...")
            d = self.d
            assert self.naive_est_flag != False
            
            x_admm = cp.zeros((self.model_d, 1))  # warm start
            z_admm = cp.zeros((self.model_d, 1))
            u_admm = cp.zeros((self.model_d, 1))
            z_new = cp.zeros((self.model_d, 1))

            mu = 1  # hyperparameter
            INV = cp.linalg.inv(mu * cp.identity(self.model_d) + self.Gamma_hat)
            est_list = []
            edge_list = []
            binarr_list = []
            lambda_list = self.lambda_list
            for l in lambda_list:
                iter_ = 10000# 1ならapproxiamte, 大きく設定したほうがexactに近い
                for _ in range(iter_):
                    x_new = INV @ (mu * (z_admm - u_admm) + self.H_hat)
                    x_dif = cp.linalg.norm(x_new - x_admm)
                    if mode=="lasso":
                        z_new = soft_threshold(l / mu, x_new + u_admm)
                    elif mode=="glasso":
                        tmp = 0
                        inc = self.single_param_dim
                        if self.single_param_dim != 0:
                            for _ in range(d):
                                z_new[tmp:tmp+inc] = shrinkage_operator(self.glasso_weight[tmp]*l / mu, x_new[tmp:tmp+inc] + u_admm[tmp:tmp+inc])
                                tmp += inc
                        inc = self.pairwise_param_dim
                        for _ in range(int(d*(d-1)/2)):
                            z_new[tmp:tmp+inc] = shrinkage_operator(self.glasso_weight[tmp]*l / mu, x_new[tmp:tmp+inc] + u_admm[tmp:tmp+inc])
                            tmp += inc
                    z_dif = cp.linalg.norm(z_new - z_admm)
                    r_dif = cp.linalg.norm(x_new - z_new)
                    u_new = u_admm + x_new - z_new
                    x_admm = x_new.copy()
                    z_admm = z_new.copy()
                    u_admm = u_new.copy()
                    if r_dif < 1e-4 and z_dif < 1e-4:
                        print(f"Finish ADMM for l = {l}")
                        break
                est_with_admm_onestep = z_admm.copy()
                est_list.append(est_with_admm_onestep)
                E = []
                B = cp.ones((self.model_d,1))
                for e in list(itertools.combinations(range(1, self.d + 1), 2)):
                    if cp.linalg.norm(self.get_param_of_vec((e[0],e[1]),est_with_admm_onestep)) > self.thresh:
                        E.append(e)
                    else:
                        ind = self.index_dictionary[e]
                        B[self.single_param_dim*d+self.pairwise_param_dim*(ind-1):self.single_param_dim*d+self.pairwise_param_dim*ind] = 0
                edge_list.append(E)
                binarr_list.append(B)
            
            def calc_SMIC(j):
                est_arr = self.naive_est
                ind_ = model_to_indices(binarr_list[j].flatten().tolist())

                #TODO: acccelerate
                # I = cp.zeros((len(ind_),len(ind_)))
                # Gamma_hat = cp.zeros((len(ind_),len(ind_)))
                # H_hat = cp.zeros((len(ind_), 1))
                # for j in range(N): #TODO: accelerate
                #     x = data[j]
                #     G_ = Gamma(x)[cp.ix_(ind_,ind_)]
                #     Gamma_hat = Gamma_hat + G_
                #     H_ = H(x)[ind_]
                #     H_hat = H_hat + H_
                #     tmp = G_ @ est_arr[ind_] - H_
                # #     I = I + tmp @ tmp.T
                # Gamma_hat = Gamma_hat/N
                # H_hat = H_hat/N
                # I = I / N
                # smic1 = N*(-est_arr[ind_].T@H_hat) 
                # smic1 = smic1.item()

                #TODO: matrix multiplications
                smic1 = n*(-est_arr[ind_].T@self.H_hat[ind_]) 
                smic1 = smic1.item()
                I_concat = cp.empty((len(ind_),n))
                for j in range(n):
                    I_concat[:,j:j+1] = Gamma(data[j])[cp.ix_(ind_,ind_)] @ est_arr[ind_] - H(data[j])[ind_]
                I = I_concat@I_concat.T
                I = cp.asnumpy(I)
                Gamma_hat = cp.asnumpy(Gamma_hat)
                eigvals = scipy.linalg.eigh(I,Gamma_hat,eigvals_only=True)
                smic2 = sum(eigvals) ### tr(IJ^-1)
                smic = smic1 + smic2 * 2
                return smic

            # scores = [calc_SMIC(j) for j in range(len(lambda_list))]
            scores = Parallel(n_jobs=-1)(delayed(calc_SMIC)(j) for j in range(len(lambda_list))) #use joblib, causes error

            opt_index = scores.index(min(scores))
            self.param = est_list[opt_index]

            ### print estimated results
            r_prev = tuple([None for _ in range(6)])
            for i in range(len(lambda_list)):
                r_new = f"Index number:{i}", lambda_list[i],scores[i], f"{len(edge_list[i])} edges", edge_list[i],est_list[i].T.tolist()[0]
                if edge_list[i] == edge_list[opt_index]:
                    print("[OPTIMAL GRAPH STRUCTURE with the smallest SMIC]")
                
                if r_new[4] == r_prev[4]:
                    pass
                else:
                    print(r_new)
                    r_prev = r_new
            
            plt.figure(figsize=(10,10))
            plt.plot([len(x) for x in edge_list],scores)
            plt.savefig(img_path)
            plt.clf()
            
            ### save results to model
            self.smic = scores
            self.est_path = est_list
            self.reg_path = edge_list
            self.bin_path = binarr_list

        else:
            pass

        end_estimation = time()
        self.param_to_graph()
        print("Estimation time(s):",end_estimation-start_estimation)
    
    def param_to_graph(self):
        for e in list(itertools.combinations(range(1, self.d + 1), 2)):
            weight = cp.linalg.norm(self.get_param((e[0],e[1])))
            if weight > self.thresh:
                self.G.add_edge(e[0],e[1],weight=weight.get().item())
            else:
                try:
                    self.G.remove_edge(e[0],e[1])
                except:
                    pass

    def graph_property(self,abbr=True):
        if not abbr:
            print("Average clustering coefficient = ", nx.average_clustering(self.G))
            print("Average shortest path length = ", nx.average_shortest_path_length(self.G))
            print("Edge number = ", len(self.G.edges))
            C = nx.community.louvain_communities(self.G, seed=123) # assume high dimensional graph
            print("Modularity = ",nx.community.modularity(self.G,C))
            print("Small-world coefficient = ", nx.sigma(self.G)) #,nx.omega(self.G))
            res = ",".join([str(x) for x in [len(self.G.edges),"{:.3f}".format(nx.community.modularity(self.G,C)),"{:.3f}".format(nx.average_clustering(self.G)),"{:.3f}".format(nx.average_shortest_path_length(self.G)),"{:.3f}".format(nx.sigma(self.G))]])
            res = "(" + res + ")"
            print(res)
        else:
            C = nx.community.louvain_communities(self.G, seed=123) # assume high dimensional graph
            res = ",".join([str(x) for x in [len(self.G.edges),"{:.3f}".format(nx.community.modularity(self.G,C)),"{:.3f}".format(nx.average_clustering(self.G)),"{:.3f}".format(nx.average_shortest_path_length(self.G))]])
            res = "(" + res + ")"
            print(res)


    def set_coordinates(self, arr):
        dic = {}
        for i in range(1,self.d+1):
            dic[i] = arr[i]
        self.pos = dic

    def plot(self, img_path="graph.png", weight=True):
        weights = nx.get_edge_attributes(self.G, 'weight').values()
        
        plt.figure(figsize=(10,10)) #グラフエリアのサイズ
        cmap=plt.cm.RdBu_r
        if weight:
            nx.draw_networkx(self.G, self.pos,node_color = 'w', edge_color = weights, edge_cmap = cmap)
        else:
            nx.draw_networkx(self.G, self.pos,node_color = 'w')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm,ax=plt.gca())
        plt.show()
        plt.savefig(img_path)

class Rotational_Model(Torus_Graph_Model):
    def __init__(self,dim):
        super().__init__(dim)
        self.single_param_dim = 2
        self.pairwise_param_dim = 2
        self.initialize()