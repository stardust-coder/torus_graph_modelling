from joblib import Parallel, delayed
from time import time
import pdb
import random
import numpy as np
import statistics
import sys
sys.path.append(".")
from model.full_model import Torus_Graph_Model
from model.rotational_model import Rotational_Model
from utils.simulation import sample_from_torus_graph
from tqdm import tqdm
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from math import sin,cos
import numba as nb
from numba import cfunc
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable
import itertools

B_list = []
B_hat_list = []
B_hat_std_list = []
M = 10**6

# for eps in [0 + 0.02*j for j in range(10)]:
for eps in [0 + 0.02]:
    # generate samples from mixtrue torus graph model
    def sample(N):
        if random.random() > eps:
            true_phi = np.array([[0,0,0,0,1,1,1,1]]).T
            data, acc = sample_from_torus_graph(N, 2, true_phi, False)
        else:
            true_phi = np.array([[0,0,0,0,5,5,5,5]]).T
            data, acc = sample_from_torus_graph(N, 2, true_phi, False)
        # print(true_phi.T, acc)
        return data

    def estimated_bias():
        N = 1000
        res = []
        for _ in tqdm(range(M)):
            data = sample(N)
            _, d = data.shape
            model = Torus_Graph_Model(d)
            model.estimate(data,mode="naive")

            I_concat = np.empty((model.model_d,N))
            for j in range(N):
                I_concat[:,j:j+1] = ((model.D_list[j])@(model.D_list[j].T @ model.naive_est) - model.H_list[j])/(N**0.5)
            I_full = I_concat@I_concat.T

            hosei = -np.trace(I_full@np.linalg.inv(model.Gamma_hat))
            res.append(hosei.item())
        return res
    
    
    
    def true_bias():
        N = 1000
        res = []
        for _ in tqdmrange(M):
            data = sample(N)
            _, d = data.shape
            model = Torus_Graph_Model(d)
            model.estimate(data,mode="naive")
            arr_ = model.param
            term1 = -arr_.T@model.H_hat
            term1 = term1.item()
            term1 *= N

            def jit_integrand_function(integrand_function):
                jitted_function = nb.njit(integrand_function, nopython=True)

                #error_model="numpy" -> Don't check for division by zero
                @cfunc(float64(intc, CPointer(float64)),error_model="numpy",fastmath=True)
                def wrapped(n, xx):
                    ar = nb.carray(xx, n)
                    return jitted_function(ar[0], ar[1])
                return LowLevelCallable(wrapped.ctypes)

            # @jit_integrand_function

            grid = 100
            xlist = [-math.pi + i * (2*math.pi/grid) + (math.pi/grid) for i in range(grid)]
            ylist = [-math.pi + i * (2*math.pi/grid) + (math.pi/grid) for i in range(grid)]
            xylist = list(itertools.product(xlist,ylist))

            def func(j):
                x,y = xylist[j]
                def mixed_bVM(x,y):
                    x_min, x_max = -math.pi,math.pi
                    def q1(x,y):
                        S = np.array([[cos(x),sin(x),cos(y),sin(y),cos(x-y),sin(x-y),cos(x+y),sin(x+y)]]).T
                        phi = np.array([[0,0,0,0,1,1,1,1]]).T
                        return np.exp(phi.T@S)
                    Z1 = integrate.dblquad(q1, x_min, x_max, x_min, x_max)[0]
                    q1_ = q1(x,y)/Z1

                    def q2(x,y):
                        S = np.array([[cos(x),sin(x),cos(y),sin(y),cos(x-y),sin(x-y),cos(x+y),sin(x+y)]]).T
                        phi = np.array([[0,0,0,0,5,5,5,5]]).T
                        return np.exp(phi.T@S)
                    Z2 = integrate.dblquad(q2, x_min, x_max, x_min, x_max)[0]
                    q2_ = q2(x,y)/Z2
                    return ((1-eps)*q1_ + eps * q2_).item()

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
                        return [[2*math.cos(x[j] - x[k])],[2*math.sin(x[j] - x[k])],[2*math.cos(x[j] + x[k])],[2*math.sin(x[j] + x[k])]]
                    def S2_2(data):
                        res = []
                        for v in itertools.combinations([i for i in range(1, len(data) + 1)], 2):
                            res.extend(S2_2_jk(data, v[0], v[1]))
                        return res
                    
                    res = S1(data)
                    res.extend(S2_2(data))
                    res = np.array(res)
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

                    for i in range(d):
                        for j in range(i+1,d):
                            v = (i,j)
                            tmp_ = [0 for _ in range(d)]
                            tmp_[v[0]] = -math.sin(x[v[0]] - x[v[1]])
                            tmp_[v[1]] = math.sin(x[v[0]] - x[v[1]])
                            entries.append(tmp_)

                            tmp_ = [0 for _ in range(d)]
                            tmp_[v[0]] = math.cos(x[v[0]] - x[v[1]])
                            tmp_[v[1]] = -math.cos(x[v[0]] - x[v[1]])
                            entries.append(tmp_)

                            tmp_ = [0 for _ in range(d)]
                            tmp_[v[0]] = -math.sin(x[v[0]] + x[v[1]])
                            tmp_[v[1]] = -math.sin(x[v[0]] + x[v[1]])
                            entries.append(tmp_)

                            tmp_ = [0 for _ in range(d)]
                            tmp_[v[0]] = math.cos(x[v[0]] + x[v[1]])
                            tmp_[v[1]] = math.cos(x[v[0]] + x[v[1]])
                            entries.append(tmp_)

                    mat_arr = np.array(entries)
                    return mat_arr

                def Gamma(x): #x : list of len(d), return mxm array
                    return D(x) @ D(x).T

                z = [x,y]
                res = 2*(arr_.T@Gamma(z)@arr_/2 - arr_.T@H(z))
                res = res.item()
                res *= mixed_bVM(x,y)
                return res
            
            # x_min, x_max = -math.pi,math.pi
            # term2 = integrate.dblquad(func, x_min, x_max, x_min, x_max)[0]
            scores = Parallel(n_jobs=-1)(delayed(func)(j) for j in range(grid**2)) #use joblib.
            term2 = sum(scores) * ((2*math.pi/grid)**2)
            term2 *= N            
            print(term1,term2,term1-term2)
            res.append(term1-term2)
        return res
    
    estimates = estimated_bias()
    B_hat = sum(estimates)/len(estimates)
    B_hat_std = statistics.pstdev(estimates)
    B_hat_list.append(B_hat)
    B_hat_std_list.append(B_hat_std)

    estimates = true_bias()
    B_true_bias = sum(estimates)/len(estimates)
    B_list.append(B_true_bias)

print(B_hat_list)
print(B_hat_std_list)
print(B_list)

import matplotlib.pyplot as plt
plt.plot([0 + 0.01*j for j in range(10)],B_hat_list)
plt.plot([0 + 0.01*j for j in range(10)],B_list)
plt.savefig("bias_correction.png")



