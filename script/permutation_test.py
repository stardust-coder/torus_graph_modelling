#Permutation Test. Shuffle the data randomly, and then perform the same estimation.
#Observe the distribution of modularity and the number of edges.

import sys
sys.path.append(".")
import numpy as np
from model.torus_graph_model import Torus_Graph_Model
from utils.gibbs import gibbs_sampler
import networkx as nx
import matplotlib.pyplot as plt

numedge_list = []
modularity_list = []

data_arr, true_edge = gibbs_sampler(N=10000) #Note: prespecified edge structures.

print("Data shape:")
print(data_arr.shape)

for _ in range(100):
    data_arr_copy = data_arr.copy()
    for j in range(data_arr_copy.shape[1]):
        np.random.shuffle(data_arr_copy[:, j])

    M = Torus_Graph_Model(19)
    M.estimate(data_arr_copy,mode="naive",img_path=f"output/tmp_naive.png")
    M.lambda_list = [0] + np.logspace(-3,0.5, num=99).tolist()
    M.glasso_weight = [0 for _ in range(2*M.d)] + [1 for _ in range(2*M.d*M.d-2*M.d)]
    for _ in range(1):
        M.estimate(data_arr_copy,mode="glasso",img_path=f"output/tmp_glasso.png")
        opt_index = M.smic.index(min(M.smic))
        if opt_index == 0 or opt_index == len(M.lambda_list)-1:
            break
        M.lambda_list = np.linspace(M.lambda_list[opt_index-1],M.lambda_list[opt_index+1],10)

    num_edge, modul = M.graph_property_min()
    numedge_list.append(num_edge)
    modularity_list.append(modul)

print("Number of edges:")
print(numedge_list)
print("Modularity")
print(modularity_list)