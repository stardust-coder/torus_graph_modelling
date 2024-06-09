import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
import pickle
from time import time
from scipy.stats import chi2
import sys
import json
import os
import pdb

sys.path.append(".")
from utils import utils, correlation
from model.model import *
import constant

# File Settings
exp_num = 26
patient_id = 26  # prefix of file names
is_simulation = False
if exp_num != 0:
    os.makedirs(f"output/{exp_num}/")
    os.makedirs(f"pickles/{exp_num}/")
ind_list, FILE_NAME_LIST = constant.get_eeg_filenames()


def main(id):
    start_time = time()

    if is_simulation:
        from simulation import sample_from_torus_graph
        import random
        FILE_NAME = "SIMULATION"
        print("Sampling might takes time ....")
        true_phi = np.zeros((50,1))
        true_phi[14:18,:] = 0.3 #(1,3)に対応
        true_phi[18:22,:] = 0.3 #(1,4)に対応
        true_phi[30:34,:] = 0.3 #(2,4)に対応
        true_phi[34:38,:] = 0.3 #(2,5)に対応
    
        print(len(true_phi)) 
        print(true_phi.T)
        data_arr, acc = sample_from_torus_graph(1000, 5, true_phi, False)

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

        # select 10 or 19 main electrodes
        montage = loaded_eeg.get_montage()
        main_electrodes = []
        # for item in ["Fp1", "Fp2", "F3", "F4"]:
        # for item in ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]:
        for item in ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T3","T4","T5","T6","Fz","Pz","Cz",]:
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
    est_dict_full = estimate_phi(data_arr)
    print("lambda:",lambda_admm_path)

    def dict_to_arr(est_d):
        return np.concatenate([x[1] for x in sorted(est_d.items())])

    est_arrs = []
    scores = []
    for j in range(len(zero_indices)):
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
        smic1 = N*(-est_arr[ind_].T@H_hat) 
        smic1 = smic1.item()
        smic2 = np.trace(I@np.linalg.inv(Gamma_hat))
        smic = smic1 + smic2
        print(smic1,"+",smic2,"=",smic)
        scores.append(smic)


    # for j in range(len(est_dict_admm_path)):
    #     print(f"Preparing {j} th result of the path...")
    #     zero_indices = 

    
    # est_dict_full = estimate_phi(data_arr)
    # import copy
    # def get_est_for_SMIC(j):  # lasso推定値→edge pattern→SM推定値を計算
    #     est_tmp = copy.deepcopy(est_dict_full)
    #     edge_list = []
    #     est = est_dict_admm_path[j]
    #     num_of_zeros = 0
    #     for k, v in est.items():
    #         if np.linalg.norm(v) <= 1e-2 and 0 not in k:
    #             est_tmp[k] = np.zeros((4, 1))
    #             num_of_zeros += 1
    #         if np.linalg.norm(v) > 1e-2 and 0 not in k:
    #             edge_list.append(k)
    #     return est_tmp, edge_list

    #     # edge_list = []
    #     # edge_list_c = []
    #     # est = est_dict_admm_path[j]
    #     # for k, v in est.items():
    #     #     if np.linalg.norm(v) > 1e-2 and 0 not in k:
    #     #         edge_list.append(k)
    #     #     if np.linalg.norm(v) <= 1e-2 and 0 not in k:
    #     #         edge_list_c.append(k)
    #     # est_tmp = estimate_phi(data_arr, invalid_edges=edge_list_c)
    #     # return est_tmp, edge_list


    # def get_SMIC(est):
    #     est_vec = [x[1] for x in sorted(est.items())]
    #     est_vec = np.concatenate(est_vec)

    #     I = np.zeros((2 * d * d, 2 * d * d))
    #     Gamma_hat = np.zeros((2 * d * d, 2 * d * d))
    #     H_hat = np.zeros((2 * d * d, 1))
    #     for j in tqdm(range(N), desc="Calculating I in SMIC", leave=False):
    #         x = data_arr[j]
    #         G_ = Gamma(x)
    #         Gamma_hat = Gamma_hat + G_
    #         H_ = H(x)
    #         H_hat = H_hat + H_
    #         tmp = G_ @ est_vec - H_
    #         I = I + tmp @ tmp.T
    #     I = I / N
    #     Gamma_hat = Gamma_hat/N
    #     H_hat = H_hat/N

    #     smic_val1 = N * (
    #         est_vec.T @ Gamma_hat @ est_vec - 2 * (est_vec.T @ H_hat)
    #     ) 
    #     smic_val1 = smic_val1.item()
    #     smic_val2 = np.trace(I @ np.linalg.inv(Gamma_hat))
    #     smic_val = smic_val1 + smic_val2
    #     print(smic_val1,"+",smic_val2,"=",smic_val)
    #     return smic_val.item()

    # edges = []
    # scores = []
    # est_dicts = []
    # for j in range(len(est_dict_admm_path)):
    #     print(f"Preparing {j} th result of the path...")
    #     est_, edge = get_est_for_SMIC(j)
    #     smic = get_SMIC(est_)
    #     scores.append(smic)
    #     edges.append(edge)
    #     est_dicts.append(est_) 
    
    
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


if __name__ == "__main__":
    for y in ind_list[patient_id]:
        main(y)
        print("Time:",y,time())
