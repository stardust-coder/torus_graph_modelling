import sys
sys.path.append(".")
from script.constant import get_eeg_filenames, get_electrode_names
from utils import utils
import mne
from mne.viz import plot_alignment, snapshot_brain_montage
import pandas as pd

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
        signal = utils.get_bandpass(signal, start, end)
        _, _, phase, _ = utils.hilbert_transform(signal=signal, verbose=verbose)
        return phase

    
    loaded_eeg = load_human_eeg(PATH_TO_DATA)
    raw_eeg = loaded_eeg.get_data()
    

    window_start = 0
    window_end = 2500
    raw_eeg = raw_eeg[:, :, window_start:window_end]
    print("Data shape:", raw_eeg.shape)

    
    df_list = []
    epochs = [0,1,2,3,4,5,6,7,8,9]  # choose which epoch
    freq_bottom = 8  # Hz
    freq_top = 14  # Hz
    num_electrodes = 91
    for epoch in epochs:
        data_df = pd.DataFrame()
        for dim_ in range(1, num_electrodes + 1):  # 91 dimensional timeseries
            data_df[f"X{dim_}"] = get_instantaneous_phase(
                raw_eeg[epoch][dim_ - 1], start=freq_bottom, end=freq_top
            )  # 2500frames, sampling=250Hz => 10 seconds
        df_list.append(data_df)

    data_df = pd.concat(df_list)
    data_arr = data_df.to_numpy()

    montage = loaded_eeg.get_montage()
    main_electrodes = []
    
    for item in get_electrode_names(dim):
        main_electrodes.append(montage.ch_names.index(item))
    
    print(
        "Index of selected electrodes(0~d)", len(main_electrodes), main_electrodes
    )
    data_arr = data_arr[:, main_electrodes]  # main electrodes
 
    return data_arr

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
    # utils.series_to_gif(raw_eeg[epoch],output_path=f"output/{exp_num}/data.gif") #raw dataを可視化

    num_electrodes = 91
    for dim_ in range(1, num_electrodes + 1):  # 91 dimensional timeseries
        data_df[f"X{dim_}"] = get_instantaneous_phase(
            raw_eeg[epoch][dim_ - 1], start=freq_bottom, end=freq_top
        )  # 2500frames, sampling=250Hz => 10 seconds
    data_arr = data_df.to_numpy()

    # select 19 or or 91 electrodes
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
    xy_pts = np.vstack([xy[ch] for ch in raw.ch_names])
    xy_pts[:,1] = -xy_pts[:,1]
    pos = xy_pts.tolist()
    return data_arr, pos