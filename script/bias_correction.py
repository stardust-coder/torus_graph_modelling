#TBD
import pdb
import random
from simulation import sample_from_torus_graph
import numpy as np
import statistics
import sys
sys.path.append(".")
from model.model import *
from tqdm import tqdm
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from math import sin,cos

B_list = []
B_hat_list = []
B_hat_std_list = []
M = 100

for eps in [0 + 0.01*j for j in range(3)]:
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
    
    #関数に投入するデータを作成
    # x = y = np.arange(-4, 4, 0.5)
    # X, Y = np.meshgrid(x, y)
    # Z = np.zeros(X.shape)
    
    # for i in range(len(x)):
    #     for j in range(len(y)):
    #         print(i,j)
    #         Z[i][j] = mixed_bVM(x[i].item(),y[j].item())
    # fig = plt.figure(figsize = (15, 15))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    # plt.show()
    # plt.savefig("bVM.png")

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

    ### Objective function
    def sm_objective_val_1(data):
        n, d = data.shape
        Gamma_hat = np.zeros((2 * d * d, 2 * d * d))
        H_hat = np.zeros((2 * d * d, 1))
        for j in range(n):
            x = data[j]
            Gamma_hat = Gamma_hat + Gamma(x)
            tmp_ = H(x)
            H_hat = H_hat + tmp_
        Gamma_hat = Gamma_hat / n
        H_hat = H_hat / n
        arr_ = estimate_phi_naive(data,verbose=True)
        
        J_emp = arr_.T@Gamma_hat@arr_/2 - arr_.T@H_hat
        J_emp = J_emp.item()


        def func(x,y):
            z = np.array([x,y])
            res = arr_.T@Gamma(z)@arr_/2 - arr_.T@H(z)
            res *= mixed_bVM(x,y)
            return res
        x_min, x_max = -math.pi,math.pi
        J_true =  integrate.dblquad(func, x_min, x_max, x_min, x_max) 

        return J_emp, J_true[0] #-0.016535077149124004, -0.01563347

    def true_bias():
        N = 1000
        
        B_ = 0
        for _ in range(M):
            data = sample(N)
            J1,J2 = sm_objective_val_1(data)
            B_ += J1 -J2
            pdb.set_trace()
        B_ = B_ / M
        B_ = B_ * 2 * N 
        return B_
    
    def estimated_bias():
        N = 1000
        res = []
        for _ in range(M):
            data = sample(N)
            n, d = data.shape
            arr_ = estimate_phi_naive(data,verbose=True)
            I = np.zeros((2*d*d,2*d*d))
            Gamma_hat = np.zeros((2*d*d,2*d*d))
            for j in range(N):
                x = data[j]
                G_ = Gamma(x)
                Gamma_hat = Gamma_hat + G_
                tmp = G_ @ arr_ - H(x)
                I = I + tmp @ tmp.T
                
            Gamma_hat = Gamma_hat/N
            I = I / N
            hosei = -np.trace(I@np.linalg.inv(Gamma_hat))
            res.append(hosei.item())
        return res

    estimates = estimated_bias()
    
    B_hat = sum(estimates)/len(estimates)
    B_hat_std = statistics.pstdev(estimates)
    B_hat_list.append(B_hat)
    B_hat_std_list.append(B_hat_std)

    B_true_bias = true_bias()
    B_list.append(B_true_bias)

print(B_hat_list)
print(B_hat_std_list)
print(B_list)

import matplotlib.pyplot as plt
plt.plot([0 + 0.01*j for j in range(20)],B_hat_list)
plt.plot([0 + 0.01*j for j in range(20)],B_list)
plt.savefig("bias_correction.png")

