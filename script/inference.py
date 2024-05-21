import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import mne
import pandas as pd
import pickle
from time import time
from scipy.stats import chi2
import sys
sys.path.append(".")
from utils import utils, correlation
from torus_graph_model.model import *

import argparse    # 1. argparseをインポート

parser = argparse.ArgumentParser()
parser.add_argument('file_name', help='PATH TO YOUR CSV FILE') 
parser.add_argument('--phase', help='Is your data circular data defined on [0,2π)?', action='store_true') 
args = parser.parse_args()  

data_df = pd.read_csv(args.file_name,header=None)
if args.phase:
    data_arr = data_df.to_numpy()
if not args.phase:
    data_df = data_df.apply(lambda x:utils.hilbert_transform(x.tolist())[2],axis=0)
    data_arr = data_df.to_numpy()

print("="*10)
print("Data shape",data_arr.shape) #shape:(N,d)
print("="*10)


exp_num = 8
FILE_NAME = "anglebeta"
import os
os.makedirs(f"output/{exp_num}/")


### save to pickle. 
start = time()

est, Gamma_zero, Gamma_zero_inv, H_zero, V_zero = estimate_phi(data_arr) # 2 seconds x N (num of frames)
#est, Gamma_zero, Gamma_zero_inv, H_zero, V_zero = estimate_phi_numpy(data_arr) # 2 seconds x N (num of frames)

print("Calculating ... ")

#Sigma_zero = Gamma_zero_inv@V_zero@Gamma_zero_inv

import cupy as cp
def ABA(A,B,C):
    return cp.dot(cp.dot(A,B),C)
print(time())
Sigma_zero_gpu = ABA(cp.asarray(Gamma_zero_inv),cp.asarray(V_zero),cp.asarray(Gamma_zero_inv))
Sigma_zero = cp.asnumpy(Sigma_zero_gpu)
print(time())

score_matching_results = {"est":est, "Gamma_zero":Gamma_zero, "Gamma_zero_inv":Gamma_zero_inv, "H_zero":H_zero, "V_zero":V_zero,"Sigma_zero":Sigma_zero}
with open(f"pickles/"+FILE_NAME+".pkl", mode="wb") as f:
    pickle.dump(score_matching_results, f)
end = time()
print("="*10)
print("Parameter estimation took", end-start, "seconds.")
print("="*10)


### load from pickle
with open("pickles/"+FILE_NAME+".pkl", mode="rb") as g:  
    score_matching_results = pickle.load(g)
    est = score_matching_results["est"]
    Sigma_zero = score_matching_results["Sigma_zero"]

n,d = data_arr.shape #n ; sample size, d : dimension
ls = [] #matrix of test stats
adj_mat = np.zeros((d,d)) #adjancy matrix of final network
alpha = 0.05


for i in range(d):
    tmp_ = []
    for j in range(d):
        if i==j:
            val = 0
        else:
            val = test_for_one_edge(n,d,est,min(i+1,j+1),max(i+1,j+1),Sigma_zero,verbose=False)
            one_minus_alpha = chi2.cdf(x = val, df = 4)            
            if 1-alpha < one_minus_alpha: #reject null hypothesis
                adj_mat[i][j] = 1
        tmp_.append(val)
    ls.append(tmp_)

corr_mat = np.array(ls)
np.fill_diagonal(corr_mat, np.max(corr_mat))
np.save(f"output/{exp_num}/{id}_test_stats_matrix",corr_mat)
np.save(f"output/{exp_num}/{id}_adjancy_matrix",adj_mat)

correlation.corr_to_corr_map(corr_mat,f"output/{exp_num}/{id}_test_stats_matrix.png")
correlation.corr_to_corr_map(adj_mat,f"output/{exp_num}/{id}_adjancy_matrix.png")

with open(f"output/{exp_num}/{id}_config.json","w") as c:
    dic = {"window_start":window_start,
                "window_end":window_end,
                "freq_bottom":freq_bottom,
                "freq_top":freq_top,
                "epoch":epoch,
                "num_electrodes":num_electrodes
                }
    json.dump(dic, c, indent=2)

# # visualize distribution of test stats
# plt.hist(corr_mat.flatten().tolist())
# plt.vlines(chi2.ppf(0.95, df = 4), 0, 5000, colors='red', linestyle='dashed', linewidth=3)
# plt.savefig("output/{exp_num}/dist_of_test_stats.png")


print("="*10)
print(f"Saved correlation matrix and result to output/{exp_num}/")
print("="*10)