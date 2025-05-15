import numpy as np
import scipy
import matplotlib.pyplot as plt
import pdb

def PLI(sig1, sig2, fs=500):
    phase_dif = sig1-sig2 #使ってない
    # f, Pxy_csd = scipy.signal.csd(sig1, sig2, fs)  
    pli = np.sign(np.sin(phase_dif))
    pli = np.abs(np.mean(pli))
    return pli 


def PLV(sig1, sig2):
    z = np.cos(sig1-sig2) + 1j * np.sin(sig1-sig2)
    z = np.mean(z)
    return np.abs(z)

# バターワースフィルタ（バンドパス）
def bandpass(x, samplerate, fp, fs=np.array([0, 6000]), gpass=3, gstop=40):
    fn = samplerate / 2  # ナイキスト周波数
    wp = fp / fn  # ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  # ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = scipy.signal.buttord(wp, ws, gpass, gstop)  # オーダーとバターワースの正規化周波数を計算
    b, a = scipy.signal.butter(N, Wn, "band")  # フィルタ伝達関数の分子と分母を計算
    y = scipy.signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける
    return y


def get_bandpass(data, start, end):
    samplerate = 1000
    fp = np.array([start, end])   # 通過域端周波数[Hz]
    fs = np.array([1, 6000])  # 阻止域端周波数[Hz]
    gpass = 3  # 通過域端最大損失[dB]
    gstop = 40  # 阻止域端最小損失[dB]
    return bandpass(data, samplerate, fp, fs, gpass, gstop)

# ヒルベルト変換
def hilbert_transform(signal, dt=1e-4, verbose=False):
    '''
    signal : np.array with shape (len,)

    Return ; 
        z :  複素数信号. zの実部はsignal, zの虚部はヒルベルト変換された信号.
        env : envelope, zの絶対値
        phase_inst : 瞬時位相
        freq_inst : 瞬時周波数
    '''

    z = scipy.signal.hilbert(signal)
    env = np.abs(z)
    phase_inst = np.angle(z)
    freq_inst = np.gradient(phase_inst)/dt/(2.0*np.pi)
    if verbose:
        plt.figure(figsize=(15, 5))
        plt.plot(signal, color='blue', label="input signal")  # 入力信号
        plt.plot(env, color='red', linestyle='dashed',
                 label="envelope")  # envelope
        plt.legend()
    return z, env, phase_inst, freq_inst

import matplotlib.animation as animation
def series_to_gif(data,output_path):
    '''
    data : numpy array (d,n)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ims = []
    for i in range(data.shape[0]):
        im = ax.plot(data[i])
        ax.set_title(f"electrode #{i+1}")
        ims.append(im)
    ani = animation.ArtistAnimation(fig,ims,interval=100,blit=True,repeat_delay=1000)
    ani.save(output_path,writer="pillow")

def dict_to_arr(est_d):
    return np.concatenate([x[1] for x in sorted(est_d.items())])


import sys
sys.path.append("..")
import pickle
import matplotlib.pyplot as plt
import itertools
import numpy as np
from script.constant import get_eeg_filenames, get_electrode_names
def draw_heatmap(M,output_img_path="heatmap.png"): #visualize estimated values in heatmap
    M1 = M.param[:2*M.d].reshape(-1,2).T
    
    # M2 = M.param[2*M.d:].reshape(-1,4).T ###for full model
    M2 = M.param[2*M.d:].reshape(-1,2).T ###for rotational model
    try:
        M1 = M1.get()
    except:
        pass

    try:
        M2 = M2.get()
    except:
        pass
    M2 = np.concatenate([M2,np.zeros(M2.shape)]) ###for rotational model
    
    fig = plt.figure(figsize=(50,5))

    ax1 = fig.add_subplot(2, 1, 1)
    cax1 = ax1.imshow(M1, aspect='auto', cmap='jet')
    fig.colorbar(cax1, ax=ax1, orientation='vertical')
    ax1.set_title('φ_i')  # Optional title
    ax1.set_xticks([i for i in range(M.d)])  # Remove x-axis ticks
    ax1.set_xticklabels([i+1 for i in range(M.d)])  # Remove x-axis ticks
    ax1.set_yticks([0,1])  # Remove y-axis ticks
    ax1.set_yticklabels(['1', '2'])  # Optionally set y-axis label

    # Second subplot
    ax2 = fig.add_subplot(2, 1, 2)
    cax2 = ax2.imshow(M2, aspect='auto',  cmap='jet')
    fig.colorbar(cax2, ax=ax2, orientation='vertical')
    ax2.set_title('φ_jk')  # Optional title
    ax2.set_xticks([i for i in range(int(M.d*(M.d-1)/2))])  # Example x-axis ticks
    # ax2.set_xticklabels([(v[0],v[1]) for v in itertools.combinations([i+1 for i in range(M.d)],2)],rotation=90)
    X = []
    for i in range(M.d-1):
        X.append(str(i+1))
        for _ in range(M.d-i-2):
            X.append("")
    ax2.set_xticklabels(X,rotation=90)
    ax2.set_yticks([0,1,2,3])  # Remove y-axis ticks
    ax2.set_yticklabels(['1', '2', '3', '4'])  # Optionally set y-axis label

    # Adjust layout
    fig.tight_layout()

    # Show the plots
    plt.show()

    # Save the figure
    fig.savefig(output_img_path)