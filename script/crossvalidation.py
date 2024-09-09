import sys
sys.path.append(".")
from model.torus_graph import Torus_Graph_Model
from utils import utils, correlation
from utils.simulation import sample_from_torus_graph, star_shaped_sample, star_shaped_rotational_sample, bagraph_sample
from observe import draw_heatmap
from data.dataloader import chennu, chennu_with_pos
from constant import get_eeg_filenames, get_electrode_names

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import itertools


def CV1():
    ###SMIC vs SMCV
    detect_prob_cv = {}
    detect_prob_ic = {}

    l = [i for i in range(1, 5 + 1)]
    for a,b in itertools.combinations(l, 2):
        detect_prob_cv[(a,b)] = 0
        detect_prob_ic[(a,b)] = 0

    for _ in tqdm(range(100)):
        data_arr = star_shaped_sample(N=100)
        M = Torus_Graph_Model(5)
        M.lambda_list = np.logspace(-2, 1, num=100).tolist()

        M.estimate(data_arr,mode="naive")
        M.glasso_weight = [0 for _ in range(2*M.d)] + [1 for _ in range(2*M.d*M.d-2*M.d)]
        M.estimate(data_arr,mode="glasso",img_path=f"smic_glasso.png")
        opt_index_smcv = M.cross_validation(data_arr)
        e_cv = M.reg_path[opt_index_smcv]
        e_ic = M.G.edges
        for a,b in itertools.combinations(l, 2):
            if (a,b) in e_cv:
                detect_prob_cv[(a,b)] = detect_prob_cv[(a,b)]+1
            if (a,b) in e_ic:
                detect_prob_ic[(a,b)] = detect_prob_ic[(a,b)]+1

    print(detect_prob_cv)
    print(detect_prob_ic)

def CV2():
    ###SMIC vs SMCV
    detect_prob_cv = {}
    detect_prob_ic = {}

    l = [i for i in range(1, 3 + 1)]
    for a,b in itertools.combinations(l, 2):
        detect_prob_cv[(a,b)] = 0
        detect_prob_ic[(a,b)] = 0

    for _ in tqdm(range(100)):
        data_arr, _ = sample_from_torus_graph(num_samples=100,d=3,phi=np.array([[0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]]).T*1, verbose=False)
        
        M = Torus_Graph_Model(3)
        M.estimate(data_arr,mode="naive")
        
        opt_index_smcv, opt_index_smic = M.cross_validation_nolasso(data_arr)
        models_ = [[(1,2)],[(1,3)],[(2,3)],[(1,2),(1,3)],[(1,2),(2,3)],[(1,3),(2,3)],[(1,2),(1,3),(2,3)]]
        e_cv = models_[opt_index_smcv]
        e_ic = models_[opt_index_smic]
        for a,b in itertools.combinations(l, 2):
            if (a,b) in e_cv:
                detect_prob_cv[(a,b)] = detect_prob_cv[(a,b)]+1
            if (a,b) in e_ic:
                detect_prob_ic[(a,b)] = detect_prob_ic[(a,b)]+1

        

    print(detect_prob_cv)
    print(detect_prob_ic)

# detect_12_cv = 0
# detect_13_cv = 0
# detect_23_cv = 0
# detect_12_ic = 0
# detect_13_ic = 0
# detect_23_ic = 0
# for _ in tqdm(range(1000)):
#     data_arr, _ = sample_from_torus_graph(num_samples=1000,d=3,phi=np.array([[0,0,0,0,0,0,0.1,0.1,0.1,0.1,0,0,0,0,0.1,0.1,0.1,0.1]]).T, verbose=False)
#     M = Torus_Graph_Model(3)
#     M.estimate(data_arr,mode="naive")
#     M.glasso_weight = [0 for _ in range(2*M.d)] + [1 for _ in range(2*M.d*M.d-2*M.d)]
#     M.estimate(data_arr,mode="glasso",img_path=f"smic_glasso.png")
#     opt_index_smcv = M.cross_validation(data_arr)
#     e_cv = M.reg_path[opt_index_smcv]
#     e_ic = M.G.edges
#     if (1,2) in e_cv:
#         detect_12_cv += 1
#     if (1,3) in e_cv:
#         detect_13_cv += 1
#     if (2,3) in e_cv:
#         detect_23_cv += 1
#     if (1,2) in e_ic:
#         detect_12_ic += 1
#     if (1,3) in e_ic:
#         detect_13_ic += 1
#     if (2,3) in e_ic:
#         detect_23_ic += 1
    
if "__main__" == __name__:
    CV1()
    # CV2()

    import pdb; pdb.set_trace()

