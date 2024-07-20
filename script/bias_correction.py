#TBD
import random
from simulation import sample_from_torus_graph
import numpy as np
import statistics
import sys
sys.path.append(".")
from model.model import *
from tqdm import tqdm


B_list = []
B_hat_list = []
B_hat_std_list = []
for eps in [0 + 0.01*j for j in range(20)]:
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
    def sm_objective_val(data):
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

        
        J_true = 0
        M = 100000
        for _ in range(M):
            z = sample(1)[0]
            J_true += arr_.T@Gamma(z)@arr_/2 - arr_.T@H(z)
        J_true /= M
        return J_emp, J_true #-0.016535077149124004, -0.01563347

    def true_bias():
        N = 1000
        M = 100000
        B_ = 0
        for _ in tqdm(range(M)):
            data = sample(N)
            J1, J2 = sm_objective_val(data)
            B_ += J1 - J2
        B_ /= M
        B_ = B_.item()
        B_ = B_ * 2 * N 
        return B_
    
    def estimated_bias():
        N = 1000
        M = 100000
        res = []
        for _ in tqdm(range(M)):
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

