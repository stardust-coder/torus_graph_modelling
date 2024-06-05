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
from model.model import *
import json
import os
import pdb


# File Settings
exp_num = 3
patient_id = 3  # prefix of file names
is_simulation = False
if exp_num != 0:
    os.makedirs(f"output/{exp_num}/")
    os.makedirs(f"pickles/{exp_num}/")

# filenames and suffix from EEG data(Chennu et al., 2016)
ind_list = {
    2: ["003", "006", "014"],
    3: ["003", "008", "021", "026"],
    5: ["004", "009", "022", "027"],
    6: ["003", "008", "013", "026"],
    7: ["003", "008", "021", "027"],
    8: ["004", "010", "015", "028"],
    9: ["003", "008", "021", "026"],
    10: ["005", "010", "015", "028"],
    13: ["003", "008", "013", "026"],
    14: ["011", "016", "031"],
    18: ["003", "009", "014", "027"],
    20: ["004", "009", "022", "027"],
    22: ["004", "009", "014"],
    23: ["003", "008", "022", "027"],
    24: ["003", "010", "015", "028"],
    25: ["003", "008", "021", "026"],
    26: ["003", "008", "013", "026"],
    27: ["001", "010", "023", "028"],
    28: ["004", "011", "016", "029"],
    29: ["005", "010", "023", "028"],
}
FILE_NAME_LIST = {
    2: "02-2010-anest 20100210 135.",
    3: "03-2010-anest 20100211 142.",
    5: "05-2010-anest 20100223 095.",
    6: "06-2010-anest 20100224 093.",
    7: "07-2010-anest 20100226 133.",
    8: "08-2010-anest 20100301 095.",
    9: "09-2010-anest 20100301 135.",
    10: "10-2010-anest 20100305 130.",
    13: "13-2010-anest 20100322 132.",
    14: "14-2010-anest 20100324 132.",
    18: "18-2010-anest 20100331 140.",
    20: "20-2010-anest 20100414 131.",
    22: "22-2010-anest 20100415 132.",
    23: "23-2010-anest 20100420 094.",
    24: "24-2010-anest 20100420 134.",
    25: "25-2010-anest 20100422 133.",
    26: "26-2010-anest 20100507 132.",
    27: "27-2010-anest 20100823 104.",
    28: "28-2010-anest 20100824 092.",
    29: "29-2010-anest 20100921 142.",
}


def main(id):
    start_time = time()

    if is_simulation:
        from simulation import sample_from_torus_graph
        import random
        print("Sampling might takes time ....")
        true_phi = np.zeros((50,1))
        true_phi[14:18,:] = 0.3 #(1,3)に対応
        true_phi[18:22,:] = 0.3 #(1,4)に対応
        true_phi[30:34,:] = 0.3 #(2,4)に対応
        true_phi[34:38,:] = 0.3 #(2,5)に対応
    
        print(len(true_phi)) 
        print(true_phi)
        data_arr, acc = sample_from_torus_graph(10000, 5, true_phi, False)

    else:
        # load human eeg data series from PlosComp journal
        FILE_NAME = f"{FILE_NAME_LIST[patient_id]}{id}"
        PATH_TO_DATA_DIR = "../../data/Sedation-RestingState/"
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
    # correlation.data_to_corr_map(data_arr,utils.PLV,f"output/{exp_num}/"+FILE_NAME+"_PLV.png")
    # plt.clf()
    # correlation.data_to_corr_map(data_arr,utils.PLI,f"output/{exp_num}/"+FILE_NAME+"_PLI.png")
    # plt.clf()

    # 2. Torus graph modelling
    temp_time = time()

    est_dict_admm_path, lambda_admm_path, G_hat, H_hat = estimate_phi_admm_path(
        data_arr
    )
    print("calculating inverse...")
    G_hat_inv = np.linalg.inv(G_hat)
    print("calculating the SM estimator without lasso...")
    est_sm_without_lasso = G_hat_inv@H_hat
    est_sm_without_lasso_dict = {}
    ind_list = list(itertools.combinations(range(1, d + 1), 2))
    for i,t in enumerate(range(1,d+1)):
        est_sm_without_lasso_dict[(0,t)] = est_sm_without_lasso[2*i:2*(i+1)]
    for i, t in tqdm(enumerate(ind_list)):
        est_sm_without_lasso_dict[(t[0], t[1])] = est_sm_without_lasso[2 * d + 4 * (i) : 2 * d + 4 * (i + 1)]
    

    import copy
    def get_est_for_SMIC(j):  # lasso推定値→edge pattern→SM推定値を計算
        est_tmp = copy.deepcopy(est_sm_without_lasso_dict)
        edge_list = []
        est = est_dict_admm_path[j]
        num_of_zeros = 0
        for k, v in est.items():
            if np.linalg.norm(v) <= 1e-2 and 0 not in k:
                est_tmp[k] = np.zeros((4, 1))
                num_of_zeros += 1
            if np.linalg.norm(v) > 1e-2 and 0 not in k:
                edge_list.append(k)
        return est_tmp, edge_list

    def get_SMIC(est):
        est_vec = [x[1] for x in sorted(est.items())]
        est_vec = np.concatenate(est_vec)

        I = np.zeros((2 * d * d, 2 * d * d))
        for ind in tqdm(range(N), desc="Calculating I in SMIC", leave=False):
            x = data_arr[ind]
            tmp = Gamma(x) @ est_vec - H(x)
            I = I + tmp @ tmp.T
        I = I / N
        smic_val1 = N * (
            est_vec.T @ G_hat @ est_vec - 2 * (est_vec.T @ H_hat)
        ) 
        smic_val2 = np.trace(I @ G_hat_inv)
        print(smic_val1,smic_val2)
        smic_val = smic_val1 + smic_val2
        return smic_val.item()

    edges = []
    scores = []
    est_dicts = []
    for j in range(len(est_dict_admm_path)):
        print(f"Preparing {j} th result of the path...")
        est_, edge = get_est_for_SMIC(j)
        smic = get_SMIC(est_)
        
        scores.append(smic)
        edges.append(edge)
        est_dicts.append(est_)
    edge_nums = [len(e) for e in edges]
    optimal_id = scores.index(min(scores))
    print(scores[optimal_id], edge_nums[optimal_id], edges[optimal_id])
    plt.plot(edge_nums, scores)
    plt.savefig("fig1.png")
    plt.clf()
    plt.plot(lambda_admm_path, scores)
    plt.xscale("log")
    plt.savefig("fig2.png")
    plt.clf()

    if not is_simulation:

        with open(f"pickles/{exp_num}/" + FILE_NAME + "_path.pkl", mode="wb") as f:
            pickle.dump(est_dicts, f)

        with open(f"pickles/{exp_num}/" + FILE_NAME + ".pkl", mode="wb") as f:
            pickle.dump(est_dicts[optimal_id], f)  # save best result

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
