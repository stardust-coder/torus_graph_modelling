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
exp_num = 3
patient_id = 3  # prefix of file names
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
    correlation.data_to_corr_map(data_arr,utils.PLV,f"output/{exp_num}/"+FILE_NAME+"_PLV.png")
    plt.clf()
    correlation.data_to_corr_map(data_arr,utils.PLI,f"output/{exp_num}/"+FILE_NAME+"_PLI.png")
    plt.clf()

    # 2. Torus graph modelling
    #est_dict= estimate_phi_naive(data_arr)
    est_mat, est_dict= estimate_phi(data_arr)

    if not is_simulation:
        with open(f"pickles/{exp_num}/" + FILE_NAME + "_mat.pkl", mode="wb") as f:
            pickle.dump(est_mat, f) 
            
        with open(f"pickles/{exp_num}/" + FILE_NAME + "_dict.pkl", mode="wb") as f:
            pickle.dump(est_dict, f)  # save one result

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
                "optimization": "naive",
            }
            json.dump(dic, c, indent=2)
    
    if is_simulation:
        pdb.set_trace()


if __name__ == "__main__":
    for y in ind_list[patient_id]:
        main(y)
