import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
import pickle
from time import time
from scipy.stats import chi2
import sys
sys.path.append(".")
from utils import utils, correlation
from torus_graph_model.model import *
#from torus_graph_model.model_cuda import * #実装失敗
import json
import pdb

exp_num = 11
import os
os.makedirs(f"output/{exp_num}/")
os.makedirs(f"pickles/{exp_num}/")

def main(id):
    start_time = time()
    # load human eeg data series from PlosComp journal
    FILE_NAME= f"29-2010-anest 20100921 142.{id}"
    PATH_TO_DATA_DIR = "data/Sedation-RestingState/"
    PATH_TO_DATA = PATH_TO_DATA_DIR + FILE_NAME + ".set"

    def load_human_eeg(input_fname, events=None):
        data = mne.io.read_epochs_eeglab(
            input_fname, verbose=False, montage_units="cm")
        return data


    def get_instantaneous_phase(signal, start, end, verbose=False):
        '''get instantaneous phase'''
        signal = utils.get_bandpass(signal, start, end)
        _, _, phase, _ = utils.hilbert_transform(signal=signal, verbose=verbose)
        return phase


    loaded_eeg = load_human_eeg(
        PATH_TO_DATA)
    raw_eeg = loaded_eeg.get_data()  # (39, 91, 2500)
    window_start = 0
    window_end = 2500
    raw_eeg = raw_eeg[:,:,window_start:window_end]
    print("Data shape:", raw_eeg.shape)

    data_df = pd.DataFrame()
    epoch = 0 # choose which epoch
    freq_bottom = 14 #Hz
    freq_top = 30 #Hz
    utils.series_to_gif(raw_eeg[epoch],output_path=f"output/{exp_num}/data.gif") #raw dataを可視化

    num_electrodes = 91
    for dim in range(1, num_electrodes+1):  # 91 dimensional timeseries
        data_df[f"X{dim}"] = get_instantaneous_phase(
            raw_eeg[epoch][dim-1], start=freq_bottom, end=freq_top)  # 2500frames, sampling=250Hz => 10 seconds
    data_arr = data_df.to_numpy()

    print("="*10)
    print("Data shape",data_arr.shape) #shape:(N,d)
    print("="*10)

    #1. Simple Correlation Matrix

    correlation.data_to_corr_map(data_arr,utils.PLV,f"output/{exp_num}/"+FILE_NAME+"_PLV.png")
    plt.clf()
    correlation.data_to_corr_map(data_arr,utils.PLI,f"output/{exp_num}/"+FILE_NAME+"_PLI.png")
    plt.clf()

    #2. Torus graph modelling

    ### save to pickle. 
    est, Gamma_zero, Gamma_zero_inv, H_zero, V_zero = estimate_phi(data_arr) # 2 seconds x N (num of frames)
    #est, Gamma_zero, Gamma_zero_inv, H_zero, V_zero = estimate_phi_numpy(data_arr) # 2 seconds x N (num of frames)
    
    temp_time = time()
    print("Calculating ... ")

    #Sigma_zero = Gamma_zero_inv@V_zero@Gamma_zero_inv

    import cupy as cp
    def ABA(A,B,C):
        return cp.dot(cp.dot(A,B),C)
    
    Sigma_zero_gpu = ABA(cp.asarray(Gamma_zero_inv),cp.asarray(V_zero),cp.asarray(Gamma_zero_inv))
    Sigma_zero = cp.asnumpy(Sigma_zero_gpu)
    

    score_matching_results = {"est":est, "Gamma_zero":Gamma_zero, "Gamma_zero_inv":Gamma_zero_inv, "H_zero":H_zero, "V_zero":V_zero,"Sigma_zero":Sigma_zero}
    with open(f"pickles/{exp_num}/"+FILE_NAME+".pkl", mode="wb") as f:
        pickle.dump(score_matching_results, f)
    
    print("="*10)
    print("Parameter estimation took", time()-temp_time, "seconds.")
    print("="*10)


    ### load from pickle
    with open(f"pickles/{exp_num}/"+FILE_NAME+".pkl", mode="rb") as g:  
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

    end_time = time()
    ###
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
                    "num_electrodes":num_electrodes,
                    "total_computation_time":end_time-start_time
                    }
        json.dump(dic, c, indent=2)

    # # visualize distribution of test stats
    # plt.hist(corr_mat.flatten().tolist())
    # plt.vlines(chi2.ppf(0.95, df = 4), 0, 5000, colors='red', linestyle='dashed', linewidth=3)
    # plt.savefig("output/{exp_num}/dist_of_test_stats.png")


    print("="*10)
    print(f"Saved correlation matrix and result to output/{exp_num}/")
    print("="*10)

if __name__=="__main__":
    main("005")
    main("010")
    main("023")
    main("028")
