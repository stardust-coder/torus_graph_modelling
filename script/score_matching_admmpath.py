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

sys.path.append(".")
from utils import utils, correlation
from model.model import *
import constant

# File Settings
exp_num = 0
patient_id = 5 # prefix of file names
is_simulation = True
if exp_num != 0:
    os.makedirs(f"output/{exp_num}/")
    os.makedirs(f"pickles/{exp_num}/")
ind_list, FILE_NAME_LIST = constant.get_eeg_filenames()


def main(id):
    start_time = time()

    if is_simulation:
        from simulation import sample_from_torus_graph
        FILE_NAME = "SIMULATION"
        print("Running simulation...")
        true_phi = np.zeros((8,1))
        true_phi[0:4,:] = 0.0 #
        true_phi[4:6,:] = 1.0 #
        true_phi[6:8,:] = 0.0 #
        
        print("True parameter:")
        print(len(true_phi)) 
        print(true_phi.T)
        data_arr, acc = sample_from_torus_graph(1000, 2, true_phi, False)

        plt.figure(figsize=(5,5))
        plt.scatter(data_arr[:,0],data_arr[:,1])
        plt.savefig("sample.png")
        plt.clf()

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
        # for item in ["Fp1","Fp2","F3","F4","C3"]:
        for item in ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T3","T4","T5","T6","Fz","Pz","Cz"]: #10-20system
        # for item in ["Fp1","E15","Fp2","E26","E23","E16","E3","E2","F7","E27","F3","E19","Fz","E4","F4","E123","F8","E39","E35","E29","E13","E6","E112","E111","E110","E115","T3","E47","E37","E31","Cz","E80","E87","E98","T4","E50","P3","E53","E54","E55","E79","E86","P4","E101","T5","E59","E60","E67","Pz","E77","E85","E91","T6","E65","E66","E72","E84","E90","O1","Oz","O2"]: #10-10system
        # for item in ['C3', 'C4', 'Cz', 'E10', 'E101', 'E102', 'E103', 'E105', 'E106', 'E109', 'E110', 'E111', 'E112', 'E115', 'E116', 'E117', 'E118', 'E12', 'E123', 'E13', 'E15', 'E16', 'E18', 'E19', 'E2', 'E20', 'E23', 'E26', 'E27', 'E28', 'E29', 'E3', 'E30', 'E31', 'E34', 'E35', 'E37', 'E39', 'E4', 'E40', 'E41', 'E42', 'E46', 'E47', 'E5', 'E50', 'E51', 'E53', 'E54', 'E55', 'E59', 'E6', 'E60', 'E61', 'E65', 'E66', 'E67', 'E7', 'E71', 'E72', 'E76', 'E77', 'E78', 'E79', 'E80', 'E84', 'E85', 'E86', 'E87', 'E90', 'E91', 'E93', 'E97', 'E98', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'Oz', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']: #all 91 electrodes
            main_electrodes.append(montage.ch_names.index(item))
        
        print(
            "Index of selected electrodes(0~d)", len(main_electrodes), main_electrodes
        )
        data_arr = data_arr[:, main_electrodes]  # main electrodes
        

    N, d = data_arr.shape
    print("=" * 10)
    print("Data shape", data_arr.shape)  # shape:(N,d)
    print("=" * 10)

    # 1. Simple Correlation Matrix
    correlation.data_to_corr_map(data_arr,utils.PLV,f"output/{exp_num}/"+FILE_NAME+"_PLV.png")
    plt.clf()
    correlation.data_to_corr_map(data_arr,utils.PLI,f"output/{exp_num}/"+FILE_NAME+"_PLI.png")
    plt.clf()

    # 2. Torus graph modelling
    est_dict_admm_path, zero_indices, non_zero_indices, edges, lambda_admm_path = estimate_phi_admm_path(data_arr)
    # est_dict_admm_path, zero_indices, non_zero_indices, edges, lambda_admm_path = estimate_phi_admm_path_parfor(data_arr) #path自体は実は使わない。
    
    est_dict_full = estimate_phi_naive(data_arr,verbose=True) #maybe need correction 
    print("lambda:",lambda_admm_path)

    def dict_to_arr(est_d):
        return np.concatenate([x[1] for x in sorted(est_d.items())])

    est_arrs = []
    scores = []

    def calc_SMIC_1():
        for j in range(len(lambda_admm_path)):
            est_arr = dict_to_arr(est_dict_full)
            ind_ = non_zero_indices[j]
            est_arr[zero_indices[j]] = 0
            est_arrs.append(est_arr)
            
            I = np.zeros((len(ind_),len(ind_)))
            Gamma_hat = np.zeros((len(ind_),len(ind_)))
            H_hat = np.zeros((len(ind_), 1))
            for j in tqdm(range(N), desc="Calculating G,H,I in SMIC", leave=False):
                x = data_arr[j]
                G_ = Gamma(x)[np.ix_(ind_,ind_)]
                Gamma_hat = Gamma_hat + G_
                H_ = H(x)[ind_]
                H_hat = H_hat + H_
                tmp = G_ @ est_arr[ind_] - H_
                I = I + tmp @ tmp.T
            I = I / N
            Gamma_hat = Gamma_hat/N
            H_hat = H_hat/N
            smic1 = N*(-est_arr[ind_].T@H_hat)  #plugged-in optimal estimator to quaratic form
            smic1 = smic1.item()

            if I.shape == (0,0):
                smic2 = 0
            else:
                eigvals = scipy.linalg.eigh(I,Gamma_hat,eigvals_only=True)
                smic2 = sum(eigvals)
            # smic2 = np.trace(I@np.linalg.inv(Gamma_hat))
            smic = smic1 + smic2
            print(smic1,"+",smic2,"=",smic)
            scores.append(smic)

    def calc_SMIC_2():
        for j in range(len(lambda_admm_path)):
            est_arr = dict_to_arr(est_dict_full)
            ind_ = non_zero_indices[j]
            est_arr[zero_indices[j]] = 0
            est_arrs.append(est_arr)
            
            I = np.zeros((2*(d**2),2*(d**2)))
            Gamma_hat = np.zeros((2*(d**2),2*(d**2)))
            H_hat = np.zeros((2*(d**2), 1))
            for j in tqdm(range(N), desc="Calculating G,H,I in SMIC", leave=False):
                x = data_arr[j]
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
            print(smic1,"+",smic2,"=",smic)
            scores.append(smic)
    
    calc_SMIC_2()

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
