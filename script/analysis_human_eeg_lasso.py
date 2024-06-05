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
import json
import os
import pdb

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


exp_num = 29
patient_id = exp_num  # prefix of file names
os.makedirs(f"output/{exp_num}/")
os.makedirs(f"pickles/{exp_num}/")


def main(id):
    start_time = time()
    # load human eeg data series from PlosComp journal
    FILE_NAME = f"{FILE_NAME_LIST[patient_id]}{id}"
    PATH_TO_DATA_DIR = "../../data/Sedation-RestingState/"
    PATH_TO_DATA = PATH_TO_DATA_DIR + FILE_NAME + ".set"

    def load_human_eeg(input_fname, events=None):
        data = mne.io.read_epochs_eeglab(input_fname, verbose=False, montage_units="cm")
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
    # for item in ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2"]:
    for item in [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Fz",
        "Pz",
        "Cz",
    ]:
        main_electrodes.append(montage.ch_names.index(item))

    print("Index of selected electrodes(0~d)", len(main_electrodes), main_electrodes)
    data_arr = data_arr[:, main_electrodes]  # main electrodes

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
    LAMBDA = 0.5
    est_dict = estimate_phi_lasso(data_arr, LAMBDA)  # 2 seconds x N (num of frames)

    print("=" * 10)
    print("Parameter estimation took", time() - temp_time, "seconds.")
    print("=" * 10)

    print(est_dict)

    with open(f"pickles/{exp_num}/" + FILE_NAME + ".pkl", mode="wb") as f:
        pickle.dump(est_dict, f)

    ### load from pickle
    # with open(f"pickles/{exp_num}/"+FILE_NAME+".pkl", mode="rb") as g:
    #    est_dict = pickle.load(g)

    with open(f"output/{exp_num}/{id}_config.json", "w") as c:
        dic = {
            "window_start": window_start,
            "window_end": window_end,
            "freq_bottom": freq_bottom,
            "freq_top": freq_top,
            "epoch": epoch,
            "num_electrodes": num_electrodes,
            "electrodes": main_electrodes,
            "lambda": LAMBDA,
            "filename": FILE_NAME,
        }
        json.dump(dic, c, indent=2)

    # # visualize distribution of test stats
    # plt.hist(corr_mat.flatten().tolist())
    # plt.vlines(chi2.ppf(0.95, df = 4), 0, 5000, colors='red', linestyle='dashed', linewidth=3)
    # plt.savefig("output/{exp_num}/dist_of_test_stats.png")

    print("=" * 10)
    print(f"Saved correlation matrix and result to output/{exp_num}/")
    print("=" * 10)

    print(time() - start_time, "seconds")


if __name__ == "__main__":
    for y in ind_list[patient_id]:
        main(y)
