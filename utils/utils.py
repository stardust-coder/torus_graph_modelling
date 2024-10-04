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