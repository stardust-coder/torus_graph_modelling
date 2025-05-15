import numpy as np
import itertools
from tqdm import tqdm
from time import time
import pdb
import matplotlib.pyplot as plt
import networkx as nx
import mne
import pickle
from scipy.stats import chi2, vonmises
import sys
import json
sys.path.append(".")
from model.torus_graph_model import *

def core_of_torus_graph(x, phi):
    """
    x : (2d^2,1)のnumpy配列, data
    phi : (2d^2,1)のnumpy配列

    Return
        (1,1)のnumpy配列
    """
    x = list(x.T[0])
    core = phi.T @ S(x)

    return np.exp(core)

def core_of_torus_graph_2(x, d, phi):
    x = list(x.T[0])
    core = phi[2*d:,:].T @ S2(x)
    return np.exp(core)

def sample_from_torus_graph(num_samples, d, phi, verbose=True):
    assert len(phi) == 2 * d * d
    def q(x):
        return 1
    phi_ = phi.flatten().tolist()  # リストに変換
    core = 0
    for ind in range(d * d):
        core += (phi_[2 * ind] ** 2 + phi_[2 * ind + 1] ** 2) ** 0.5

    k = np.exp(core)  # 上界
    if verbose:
        print("upper bound constant", k)
    samples = []
    trial = 0
    acceptance = 0
    reject = 0

    while acceptance < num_samples:
        trial += 1
        # 提案分布からサンプリング. この例では一様.
        x = np.random.random_sample((d, 1)) * 2 * np.pi
        p = core_of_torus_graph(x, phi)
        u = np.random.random_sample()
        if u <= p / (k * q(x)):  # accept
            samples.append(x)
            acceptance += 1
            if verbose:
                print(f"{len(samples)}/{num_samples}")
        else:  # reject
            reject += 1
    if verbose:
        print("acceptance rate:", acceptance / trial)
    return np.concatenate(samples, axis=1).T, acceptance / trial


def torus_graph_density(phi, x1, x2):
    kernel = 0
    kernel += phi[0] * np.cos(x1) + phi[1] * np.sin(x1)
    kernel += phi[2] * np.cos(x2) + phi[3] * np.sin(x2)
    kernel += (
        phi[4] * np.cos(x1 - x2)
        + phi[5] * np.sin(x1 - x2)
        + phi[6] * np.cos(x1 + x2)
        + phi[7] * np.sin(x1 + x2)
    )
    return np.exp(kernel)

def star_shaped_sample(N):
    true_phi = np.zeros((50,1))
    true_phi[0:10,:] = 0.0 #
    true_phi[14:18,:] = 0.3 #(1,3)に対応
    true_phi[18:22,:] = 0.3 #(1,4)に対応
    true_phi[30:34,:] = 0.3 #(2,4)に対応
    true_phi[34:38,:] = 0.3 #(2,5)に対応
    true_phi[42:46,:] = 0.3 #(3,5)に対応
    data_arr, acc = sample_from_torus_graph(N, 5, true_phi, False)
    return data_arr

def star_shaped_rotational_sample(N):
    true_phi = np.zeros((50,1))
    true_phi[0:10,:] = 0.0 #
    true_phi[14:16,:] = 0.3 #(1,3)に対応
    true_phi[18:20,:] = 0.3 #(1,4)に対応
    true_phi[30:32,:] = 0.3 #(2,4)に対応
    true_phi[34:36,:] = 0.3 #(2,5)に対応
    true_phi[42:44,:] = 0.3 #(3,5)に対応
    data_arr, acc = sample_from_torus_graph(N, 5, true_phi, False)
    return data_arr

def bagraph_sample(N):
    d = 61
    m = 5
    G_ = nx.barabasi_albert_graph(d, m, seed=None, initial_graph=None)
    true_phi = np.zeros((2*d*d,1))

    index_dictionary = {}
    for i, v in enumerate(list(itertools.combinations(range(0, d), 2))):
        index_dictionary[v] = i
    
    for e in G_.edges:
        ind = index_dictionary[e]
        true_phi[2*d+4*ind:2*d+4*ind+4] = 0.1
    
    data_arr, acc = sample_from_torus_graph(N, d, true_phi, False)
    
    return data_arr

    


    
    
    

