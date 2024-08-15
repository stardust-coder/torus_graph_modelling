import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
import pickle
from time import time
import scipy
from scipy.stats import chi2
import sys
import json
import os
import pdb
import random
from constant import get_eeg_filenames, get_electrode_names
from parfor import parfor

sys.path.append(".")
from utils import utils, correlation
from model.model import *
import constant

# File Settings
exp_num = 0
patient_id = 6 # prefix of file names
is_simulation = False
# if exp_num != 0:
#     os.makedirs(f"output/{exp_num}/")
#     os.makedirs(f"pickles/{exp_num}/")
ind_list, FILE_NAME_LIST = get_eeg_filenames()


def main(id):
    start_time = time()

    if is_simulation:
        from simulation import sample_from_torus_graph
        FILE_NAME = "SIMULATION"
        print("Running simulation...")
        true_phi = np.zeros((50,1))
        true_phi[0:10,:] = 0.0 #
        true_phi[14:18,:] = 0.3 #(1,3)に対応
        true_phi[18:22,:] = 0.3 #(1,4)に対応
        true_phi[30:34,:] = 0.3 #(2,4)に対応
        true_phi[34:38,:] = 0.3 #(2,5)に対応
        true_phi[42:46,:] = 0.3 #(3,5)に対応
        print("True parameter:")
        print(len(true_phi)) 
        print(true_phi.T)
        data_arr, acc = sample_from_torus_graph(10000, 5, true_phi, False)
        

    else:
        # load human eeg data series from PlosComp journal
        FILE_NAME = f"{FILE_NAME_LIST[patient_id]}{id}"
        PATH_TO_DATA_DIR = "../data/Sedation-RestingState/"
        PATH_TO_DATA = PATH_TO_DATA_DIR + FILE_NAME + ".set"

        def load_human_eeg(input_fname, events=None):
            data = mne.io.read_epochs_eeglab(
                input_fname, verbose=False, montage_units="cm"
            )
            return data

        def get_instantaneous_phase(signal, start, end, verbose=False):
            """get instantaneous phase"""
            signal = utils.get_bandpass(signal, start, end)
            _, _, phase, _ = utils.hilbert_transform(signal=signal, verbose=verbose)
            return phase

        loaded_eeg = load_human_eeg(PATH_TO_DATA)
        raw_eeg = loaded_eeg.get_data()  # (39, 91, 2500)
        window_start = 0
        window_end = 2500
        raw_eeg = raw_eeg[:, :, window_start:window_end]
        print("Data shape:", raw_eeg.shape)

        data_df = pd.DataFrame()
        epoch = 0  # choose which epoch
        freq_bottom = 8  # Hz
        freq_top = 14  # Hz
        # utils.series_to_gif(raw_eeg[epoch],output_path=f"output/{exp_num}/data.gif") #raw dataを可視化

        num_electrodes = 91
        for dim in range(1, num_electrodes + 1):  # 91 dimensional timeseries
            data_df[f"X{dim}"] = get_instantaneous_phase(
                raw_eeg[epoch][dim - 1], start=freq_bottom, end=freq_top
            )  # 2500frames, sampling=250Hz => 10 seconds
        data_arr = data_df.to_numpy()

        # select 19 or or 91 electrodes
        montage = loaded_eeg.get_montage()
        main_electrodes = []
        for item in get_electrode_names(19):
            main_electrodes.append(montage.ch_names.index(item))
        
        print(
            "Index of selected electrodes(0~d)", len(main_electrodes), main_electrodes
        )
        data_arr = data_arr[:, main_electrodes]  # main electrodes
        

    N, d = data_arr.shape
    print("=" * 10)
    print("Data shape", data_arr.shape)  # shape:(N,d)
    print("=" * 10)

    ### 1. Simple Correlation Matrix
    # correlation.data_to_corr_map(data_arr,utils.PLV,f"output/{exp_num}/"+FILE_NAME+"_PLV.png")
    # plt.clf()
    # correlation.data_to_corr_map(data_arr,utils.PLI,f"output/{exp_num}/"+FILE_NAME+"_PLI.png")
    # plt.clf()

    ### 2. Torus graph modelling
    est_dict_admm_path, edges, bin_arrs, lambda_admm_path = estimate_phi_naive_admm_path(data_arr) 
    est_dict_full = estimate_phi_naive(data_arr) 

    print("lambda:",lambda_admm_path)
    # import pdb; pdb.set_trace()

    # @parfor(range(len(lambda_admm_path)))
    def calc_SMIC(j):
        est_arr = utils.dict_to_arr(est_dict_full)
        est_arr = est_arr * bin_arrs[j]
        I = np.zeros((2*(d**2),2*(d**2)))
        Gamma_hat = np.zeros((2*(d**2),2*(d**2)))
        H_hat = np.zeros((2*(d**2), 1))
        for data_ind in range(N):
            x = data_arr[data_ind]
            G_ = Gamma(x)
            Gamma_hat = Gamma_hat + G_
            H_ = H(x)
            H_hat = H_hat + H_
            tmp = G_ @ est_arr - H_
            I = I + tmp @ tmp.T
        I = I / N
        Gamma_hat = Gamma_hat/N
        H_hat = H_hat/N
        smic1 = N*(-est_arr.T@H_hat)  #plugged-in optimal estimator to quaratic form
        smic1 = smic1.item()

        eigvals = scipy.linalg.eigh(I,Gamma_hat,eigvals_only=True)
        smic2 = sum(eigvals)
        smic = smic1 + smic2
        return smic

    scores = []
    for j in tqdm(range(30)):
        scores.append(calc_SMIC(j))

    # scores = calc_SMIC
    print(scores)
    import pdb; pdb.set_trace()

    num_edges = [len(e) for e in edges]
    optimal_id = scores.index(min(scores))
    print(scores[optimal_id], num_edges[optimal_id], edges[optimal_id])
    plt.plot(num_edges, scores)
    plt.savefig(f"output/{exp_num}/" + FILE_NAME + "_fig1.png")
    plt.clf()
    plt.plot(lambda_admm_path, scores)
    plt.xscale("log")
    plt.savefig(f"output/{exp_num}/" + FILE_NAME + "_fig2.png")
    plt.clf()

    if not is_simulation:
        with open(f"pickles/{exp_num}/" + FILE_NAME + "_path.pkl", mode="wb") as f:
            pickle.dump(est_dict_admm_path, f)

        with open(f"pickles/{exp_num}/" + FILE_NAME + "_best.pkl", mode="wb") as f:
            pickle.dump(est_dict_admm_path[optimal_id], f)  # save best result
        
        with open(f"pickles/{exp_num}/" + FILE_NAME + "_smic.pkl", mode="wb") as f:
            pickle.dump(scores, f)  # save smic

        with open(f"output/{exp_num}/{id}_config.json", "w") as c:
            dic = {
                "window_start": window_start,
                "window_end": window_end,
                "freq_bottom": freq_bottom,
                "freq_top": freq_top,
                "epoch": epoch,
                "num_electrodes": num_electrodes,
                "electrodes": main_electrodes,
                "filename": FILE_NAME,
                "optimization": "admm_onestep",
            }
            json.dump(dic, c, indent=2)
    
    if is_simulation:
        pdb.set_trace()
        print(num_edges)
        print(edges)
        print(scores)

    # constuct a graph
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(edges[optimal_id])
    print(G)
    print("Average clustering coefficient = ", nx.average_clustering(G))
    print("Average shortest path length = ", nx.average_shortest_path_length(G))
    print("Small-world index = ", nx.sigma(G),nx.omega(G))

if __name__ == "__main__":
    for y in ind_list[patient_id]:
        main(y)
        print("Time:",y,time())
