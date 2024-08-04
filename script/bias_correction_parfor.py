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
from parfor import parfor

true_phi_1 = np.array([[0,0,0,0,1,1,1,1]]).T
true_phi_2 = np.array([[0,0,0,0,0.1,0.1,0.1,0.1]]).T

x_min, x_max = 0,2*math.pi
def q1(x,y):
    S = np.array([[cos(x),sin(x),cos(y),sin(y),cos(x-y),sin(x-y),cos(x+y),sin(x+y)]]).T
    return np.exp(true_phi_1.T@S)
Z1 = integrate.dblquad(q1, x_min, x_max, x_min, x_max)[0]

def q2(x,y):
    S = np.array([[cos(x),sin(x),cos(y),sin(y),cos(x-y),sin(x-y),cos(x+y),sin(x+y)]]).T
    return np.exp(true_phi_2.T@S)
Z2 = integrate.dblquad(q2, x_min, x_max, x_min, x_max)[0]


def func2():
    @parfor([0 + 0.01*j for j in range(30)])
    def true_bias(eps):
        def mixed_bVM(x,y):
            q1_ = q1(x,y)/Z1
            q2_ = q2(x,y)/Z2
            return ((1-eps)*q1_ + eps * q2_).item()

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
            arr_ = np.linalg.inv(Gamma_hat)@H_hat
            
            J_emp = arr_.T@Gamma_hat@arr_/2 - arr_.T@H_hat
            J_emp = J_emp.item()

            def func(x,y):
                z = np.array([x,y])
                res = arr_.T@Gamma(z)@arr_/2 - arr_.T@H(z)
                res *= mixed_bVM(x,y)
                return res.item()

            J_true = integrate.dblquad(func, 0, 2*math.pi, 0, 2*math.pi)
            
            return J_emp, J_true[0]
    
        def sample(N):
            dataset = []
            for _ in range(N):
                if random.random() > eps:
                    data, acc = sample_from_torus_graph(1, 2, true_phi_1, False)
                else:
                    data, acc = sample_from_torus_graph(1, 2, true_phi_2, False)
                dataset.append(data)
            data_arr = np.concatenate(dataset)
            return data_arr

        N = 1000
        M = 100000
        B_ = []

        for k in range(M):
        # for k in tqdm(range(M)):
            data = sample(N)
            J1,J2 = sm_objective_val_1(data)
            B_.append(N*(J1 - J2))

        B_mean = sum(B_)/len(B_)
        B_std = statistics.pstdev(B_)
        
        return B_mean, B_std
    
    l = true_bias
    print(l)

def func1():
    @parfor([0 + 0.01*j for j in range(30)])
    def estimated_bias(eps):
        def sample(N):
            dataset = []
            for _ in range(N):
                if random.random() > eps:
                    data, acc = sample_from_torus_graph(1, 2, true_phi_1, False)
                else:
                    data, acc = sample_from_torus_graph(1, 2, true_phi_2, False)
                dataset.append(data)
            data_arr = np.concatenate(dataset)
            return data_arr

        N = 1000
        M = 100000
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
        return sum(res)/len(res), statistics.pstdev(res)
    
    l = estimated_bias
    print(l)

if __name__ == "__main__":
    func1()
    # for e in [0 + 0.01*j for j in range(30)]:
    #     func2(e)
    func2()



# B_hat_list = estimated_bias
# print(B_hat_list)


# import matplotlib.pyplot as plt
# plt.plot([0 + 0.01*j for j in range(20)],B_hat_list)
# plt.plot([0 + 0.01*j for j in range(20)],B_list)
# plt.savefig("bias_correction.png")

