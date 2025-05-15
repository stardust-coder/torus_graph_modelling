import sys
sys.path.append(".")
from script.constant import get_eeg_filenames, get_electrode_names
from utils import utils
import mne
from mne.viz import plot_alignment, snapshot_brain_montage
import pandas as pd
import scipy
import numpy as np

ind_list, FILE_NAME_LIST = get_eeg_filenames()

def chennu(patient_id,state_id,dim):
    FILE_NAME = f"{FILE_NAME_LIST[patient_id]}{state_id}"
    PATH_TO_DATA_DIR = "../data/Sedation-RestingState/"
    PATH_TO_DATA = PATH_TO_DATA_DIR + FILE_NAME + ".set"

    def load_human_eeg(input_fname, events=None):
        data = mne.io.read_epochs_eeglab(
            input_fname, verbose=False, montage_units="cm"
        )
        return data

    def get_instantaneous_phase(signal, start, end, verbose=False):
        """get instantaneous phase"""
        signal_ = utils.get_bandpass(signal, start, end)
        _, _, phase, _ = utils.hilbert_transform(signal=signal_, verbose=verbose)
        return phase

    def get_envelope(signal, start, end, verbose=False):
        """get amplitude of power"""
        signal_ = utils.get_bandpass(signal, start, end)
        _, env, _, _ = utils.hilbert_transform(signal=signal_, verbose=verbose)
        return env
    
    loaded_eeg = load_human_eeg(PATH_TO_DATA)
    raw_eeg = loaded_eeg.get_data()
    
    ### Phase
    window_start = 0
    window_end = 2500
    raw_eeg = raw_eeg[:, :, window_start:window_end]
    epochs = [0,1,2,3,4,5,6,7,8,9]  # choose which epoch
    freq_bottom = 8  # Hz
    freq_top = 15  # Hz α：8-15, β: 12-25, γ: 25-40
    num_electrodes = 91

    df_list = []
    for epoch in epochs:
        data_df = pd.DataFrame()
        for dim_ in range(1, num_electrodes + 1):  # 91 dimensional timeseries
            data_df[f"X{dim_}"] = get_instantaneous_phase(
                raw_eeg[epoch][dim_ - 1], start=freq_bottom, end=freq_top
            )  # 2500frames, sampling=250Hz => 10 seconds
        df_list.append(data_df)
    data_df = pd.concat(df_list)
    phase_data_arr = data_df.to_numpy()

    ### Amplitude
    window_start = 0
    window_end = 2500
    raw_eeg = raw_eeg[:, :, window_start:window_end]
    epochs = [0,1,2,3,4,5,6,7,8,9]  # choose which epoch
    freq_bottom = 25  # Hz
    freq_top = 40  # Hz α：8-15, β: 12-25, γ: 25-40
    num_electrodes = 91
    df_list = []
    for epoch in epochs:
        data_df = pd.DataFrame()
        for dim_ in range(1, num_electrodes + 1):  # 91 dimensional timeseries
            data_df[f"X{dim_}"] = get_envelope(
                raw_eeg[epoch][dim_ - 1], start=freq_bottom, end=freq_top
            )  # 2500frames, sampling=250Hz => 10 seconds
        df_list.append(data_df)
    data_df = pd.concat(df_list)
    power_data_arr = data_df.to_numpy()

    ### Extract target electrode channels
    montage = loaded_eeg.get_montage()
    main_electrodes = []
    
    for item in get_electrode_names(dim): #choose dim from {5,9,61,91}
        main_electrodes.append(montage.ch_names.index(item))
    
    print(
        "Index of selected electrodes(0~d)", len(main_electrodes), main_electrodes
    )
    phase_data_arr = phase_data_arr[:, main_electrodes] 
    power_data_arr = power_data_arr[:, main_electrodes]
    return phase_data_arr, power_data_arr


def chennu_for_debug(patient_id,state_id,dim):
    FILE_NAME = f"{FILE_NAME_LIST[patient_id]}{state_id}"
    PATH_TO_DATA_DIR = "../data/Sedation-RestingState/"
    PATH_TO_DATA = PATH_TO_DATA_DIR + FILE_NAME + ".set"

    def load_human_eeg(input_fname, events=None):
        data = mne.io.read_epochs_eeglab(
            input_fname, verbose=False, montage_units="cm"
        )
        return data

    def get_instantaneous_phase(signal, start, end, verbose=False):
        """get instantaneous phase"""
        signal_ = utils.get_bandpass(signal, start, end)
        _, _, phase, _ = utils.hilbert_transform(signal=signal_, verbose=verbose)
        return phase

    def get_envelope(signal, start, end, verbose=False):
        """get amplitude of power"""
        signal_ = utils.get_bandpass(signal, start, end)
        _, env, _, _ = utils.hilbert_transform(signal=signal_, verbose=verbose)
        return env
    
    loaded_eeg = load_human_eeg(PATH_TO_DATA)
    raw_eeg = loaded_eeg.get_data()
    
    '''
    ### Settings
    '''
    window_start = 0
    window_end = 2500
    raw_eeg = raw_eeg[:, :, window_start:window_end]
    epochs = [0,1,2,3,4,5,6,7,8,9]  # choose which epoch
    num_electrodes = 91

    '''
    ### Data preprocessing
    '''
    ###Phase
    df_list = []
    for epoch in epochs:
        data_df = pd.DataFrame()
        for dim_ in range(1, num_electrodes + 1):  # 91 dimensional timeseries
            data_df[f"X{dim_}"] = get_instantaneous_phase(
                raw_eeg[epoch][dim_ - 1], start=8, end=15
            )  # 2500frames, sampling=250Hz => 10 seconds
        df_list.append(data_df)
    data_df = pd.concat(df_list)
    phase_data_arr = data_df.to_numpy()
    
    ###Amplitude
    df_list = []
    for epoch in epochs:
        data_df = pd.DataFrame()
        for dim_ in range(1, num_electrodes + 1):  # 91 dimensional timeseries
            data_df[f"X{dim_}"] = get_envelope(
                raw_eeg[epoch][dim_ - 1], start=8, end=15
            )  # 2500frames, sampling=250Hz => 10 seconds
        df_list.append(data_df)
    data_df = pd.concat(df_list)
    power_data_arr = data_df.to_numpy()

    ###Raw
    df_list = []
    for epoch in epochs:
        data_df = pd.DataFrame()
        for dim_ in range(1, num_electrodes + 1):  # 91 dimensional timeseries
            data_df[f"X{dim_}"] = raw_eeg[epoch][dim_ - 1]
        df_list.append(data_df)
    data_df = pd.concat(df_list)
    raw_data_arr = data_df.to_numpy()

    ###Alpha/Beta/Gammas
    df_list = []
    for epoch in epochs:
        data_df = pd.DataFrame()
        for dim_ in range(1, num_electrodes + 1):  # 91 dimensional timeseries
            data_df[f"X{dim_}"] = utils.get_bandpass(
                raw_eeg[epoch][dim_ - 1], start=8, end=15
            ) 
        df_list.append(data_df)
    data_df = pd.concat(df_list)
    band_data_arr = data_df.to_numpy()

    ### Extract target electrode channels
    montage = loaded_eeg.get_montage()
    main_electrodes = []
    
    for item in get_electrode_names(dim): #choose dim from {5,9,61,91}
        main_electrodes.append(montage.ch_names.index(item))
    
    print(
        "Index of selected electrodes(0~d)", len(main_electrodes), main_electrodes
    )
    phase_data_arr = phase_data_arr[:, main_electrodes] 
    power_data_arr = power_data_arr[:, main_electrodes]
    raw_data_arr = raw_data_arr[:, main_electrodes]
    band_data_arr = band_data_arr[:, main_electrodes]
    
    return phase_data_arr, power_data_arr, raw_data_arr, band_data_arr

def chennu_with_pos(patient_id,state_id,dim):
    FILE_NAME = f"{FILE_NAME_LIST[patient_id]}{state_id}"
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
    raw_eeg = loaded_eeg.get_data() 
    
    window_start = 0
    window_end = 2500
    raw_eeg = raw_eeg[:, :, window_start:window_end]
    print("Data shape:", raw_eeg.shape)

    data_df = pd.DataFrame()
    epoch = 0  # choose which epoch
    freq_bottom = 8  # Hz
    freq_top = 14  # Hz

    num_electrodes = 91
    for dim_ in range(1, num_electrodes + 1):  # 91 dimensional timeseries
        data_df[f"X{dim_}"] = get_instantaneous_phase(
            raw_eeg[epoch][dim_ - 1], start=freq_bottom, end=freq_top
        )  # 2500frames, sampling=250Hz => 10 seconds
    data_arr = data_df.to_numpy()

    montage = loaded_eeg.get_montage()
    main_electrodes = []
    
    for item in get_electrode_names(dim):
        main_electrodes.append(montage.ch_names.index(item))
    
    print(
        "Index of selected electrodes(0~d)", len(main_electrodes), main_electrodes
    )
    data_arr = data_arr[:, main_electrodes]  # main electrodes
    
    mne.viz.set_3d_backend("notebook")
    mne.viz.set_browser_backend("matplotlib")
    fig = plot_alignment(loaded_eeg.info)
    xy, im = snapshot_brain_montage(fig,montage)
    xy_pts = np.vstack([xy[ch] for ch in raw_eeg.ch_names])
    xy_pts[:,1] = -xy_pts[:,1]
    pos = xy_pts.tolist()
    return data_arr, pos

def marmoset_ecog(name,ind):
    assert name in ["Ji", "Or"]

    def load_marmoset_ecog():
        length = -1
        if name == "Ji":
            assert ind in [1,2,3,5,15]
            res = []
            for chan in range(1,97):
                data = scipy.io.loadmat(f'../data/riken-auditory-ECoG/Ji20180308S{ind}c/ECoG_ch{chan}.mat')['ECoGData'][:,:length]
                res.append(data)
            return np.concatenate(res,axis=0)
        if name == "Or":
            assert ind in [2,3,4,6,16]
            res = []
            for chan in range(1,97):
                data = scipy.io.loadmat(f'../data/riken-auditory-ECoG/Or20171207S{ind}c/ECoG_ch{chan}.mat')['ECoGData'][:,:length]
                res.append(data)
            return np.concatenate(res,axis=0)

    def get_instantaneous_phase(signal, start, end, verbose=False):
        """get instantaneous phase"""
        signal_ = utils.get_bandpass(signal, start, end)
        _, _, phase, _ = utils.hilbert_transform(signal=signal_, verbose=verbose)
        return phase
    
    '''
    ### Settings
    '''
    window_start = 0
    window_end = 25000
    raw_ecog = load_marmoset_ecog()
    raw_ecog = raw_ecog[:, window_start:window_end]
    freq_bottom = 25  # Hz
    freq_top = 40  # Hz α：8-15, β: 12-25, γ: 25-40
    num_electrodes = 96

    '''
    ### Data preprocessing
    '''
    ###Phase
    df_list = []
    data_df = pd.DataFrame()
    for dim_ in range(1, num_electrodes + 1):  # 96 dimensional timeseries
        data_df[f"X{dim_}"] = get_instantaneous_phase(
            raw_ecog[dim_ - 1], start=freq_bottom, end=freq_top
        )  
    df_list.append(data_df)
    data_df = pd.concat(df_list)
    phase_data_arr = data_df.to_numpy()
    return phase_data_arr