import sys
sys.path.append(".")
import copy
from math import cos, sin
import random
import numpy as np
from scipy.stats import vonmises
import itertools
import math
from utils.simulation import sample_from_torus_graph
from tqdm import tqdm


def x_minus_k(x,k):
    y = copy.deepcopy(x)
    y.pop(k)
    return y

def gibbs_sample(phi,data,k):
    d = len(data)

    phi_1 = phi[:2*d]
    phi_2 = phi[2*d:]
    kappa = [np.sqrt(phi_1[2*j:2*j+1][0]**2 + phi_1[2*j+1:2*j+2][0]**2) for j in range(d)]
    mu = [np.arctan2(phi_1[2*j+1:2*j+2][0],phi_1[2*j:2*j+1][0]) for j in range(d)]
    
    #k : 除去するインデックス。0-indexなので注意.
    x = data
    L = [kappa[k]]
    alpha = phi[2*d::4]
    beta = phi[2*d+1::4]
    gamma = phi[2*d+2::4]
    delta = phi[2*d+3::4]
    comb = list(itertools.combinations([i for i in range(1,d+1)],2))
    extracted = []
    for i,item in enumerate(comb):
        if k+1 in [item[0],item[1]]:
            extracted.append(i)

    L = L + [alpha[i] for i in extracted]
    L = L + [beta[i] for i in extracted]
    L = L + [gamma[i] for i in extracted]
    L = L + [delta[i] for i in extracted] #len:241 = 1 + 4 * (61-1)


    V = [mu[k]]
    x_k = x_minus_k(x,k)
    V = V + x_k
    tmp = copy.deepcopy(x_k)
    for i in range(len(tmp)):
        if i < k:
            tmp[i] -= math.pi/2
        else:
            tmp[i] += math.pi/2
    V = V + tmp
    tmp = [-x for x in x_k]
    V = V + tmp
    tmp = [item + math.pi/2 for item in tmp]
    V = V + tmp #len: 241
    

    bx = sum([L[m]*cos(V[m]) for m in range(len(V))])
    by = sum([L[m]*sin(V[m]) for m in range(len(V))])
    A = (bx**2+by**2)**0.5
    Delta = np.arctan2(by,bx)

    loc = Delta
    kappa = A
    sample_size = 1
    
    sample = vonmises(loc=loc, kappa=kappa).rvs(sample_size)
    return sample.item()

def gibbs_sampler(N):
    d = 19
    comb = list(itertools.combinations([i for i in range(1,d+1)],2))
    selected_edges = []
    #random vector for model parameters
    phi = []
    for i in range(d):
        phi.extend([0.0,0.0])
    for i in range(int(d*(d-1)/2)):
        if random.random() < 0.5:
            phi.extend([0.3,0.3,0.0,0.0])
            selected_edges.append(comb[i])
        else:
            phi.extend([0,0,0,0])
        
    #Rejection sampling
    # samples_rj, _ = sample_from_torus_graph(N,d,np.array([phi]).T,verbose=False)
    # samples_rj = samples_rj.tolist()

    ###Gibbs sampling
    samples_gbs = []
    x = [random.random()*math.pi*2 for _ in range(d)] # init data
    
    burnin = 10000
    for j in tqdm(range(N+burnin)):
        for k in range(d):
            sample = gibbs_sample(phi,x,k)
            x[k] = sample
        
        if j >= burnin:
            samples_gbs.append(copy.deepcopy(x))

    # return samples_rj, samples_gbs # for debug
    return np.array(samples_gbs), selected_edges


if __name__ == "__main__":
    samples_rj, samples_gbs = gibbs_sampler()

    import matplotlib.pyplot as plt
    def adjust(x):
        if x < 0:
            return x + 2*math.pi
        if x > 2*math.pi:
            return x - 2*math.pi
        else:
            return x

    x = [adjust(item[0]) for item in samples_rj]
    y = [adjust(item[1]) for item in samples_rj]
    plt.figure(figsize=(5,5))
    plt.scatter(x,y)
    plt.savefig("sample_rj_data.png")

    plt.clf()
    x = [adjust(item[0]) for item in samples_gbs]
    y = [adjust(item[1]) for item in samples_gbs]
    plt.figure(figsize=(5,5))
    plt.scatter(x,y)
    plt.savefig("sample_gbs_data.png")