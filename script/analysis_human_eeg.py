import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
import pickle
from scipy.stats import chi2
import sys
sys.path.append(".")
from utils import utils, correlation
from torus_graph_model.model import *

# load human eeg data series from PlosComp journal
FILE_NAME= "03-2010-anest 20100211 142.021.set"
PATH_TO_DATA_DIR = "../../data/Sedation-RestingState/"
PATH_TO_DATA = PATH_TO_DATA_DIR + FILE_NAME

def load_human_eeg(input_fname, events=None):
    data = mne.io.read_epochs_eeglab(
        input_fname, verbose=False, montage_units="cm")
    return data


def get_instantaneous_phase(signal, start, end, verbose=False):
    '''get instantaneous phase'''
    signal = utils.get_bandpass(signal, start, end)
    _, _, phase, _ = utils.hilbert_transform(signal=signal, verbose=verbose)
    return phase


loaded_eeg = load_human_eeg(
    PATH_TO_DATA)
raw_eeg = loaded_eeg.get_data()  # (39, 91, 2500)


data_df = pd.DataFrame()
epoch = 0
start = 10 #Hz
end = 15 #Hz

for dim in range(1, 92):  # 91 dimensional timeseries
    data_df[f"X{dim}"] = get_instantaneous_phase(
        raw_eeg[epoch][dim-1], start=start, end=end)  # 2500frames, sampling=250Hz => 10 seconds
data_arr = data_df.to_numpy()

print("="*10)
print("Data shape",data_arr.shape)
print("="*10)

#Simple Correlation Matrix

#correlation.data_to_corr_map(data_arr,utils.PLV)
#correlation.data_to_corr_map(data_arr,utils.PLI)

#Torus graph modelling

### save to pickle. #TODO: needs acceleration by parallel or numpy. this computation takes time.
# est, Gamma_zero, Gamma_zero_inv, H_zero, V_zero = estimate_phi(data_arr)
# Sigma_zero = Gamma_zero_inv@V_zero@Gamma_zero_inv
# score_matching_results = {"est":est, "Gamma_zero":Gamma_zero, "Gamma_zero_inv":Gamma_zero_inv, "H_zero":H_zero, "V_zero":V_zero,"Sigma_zero":Sigma_zero}
# with open(f"pickles/"+FILE_NAME, mode="wb") as f:
#     pickle.dump(score_matching_results, f)

### load from pickle
with open("pickles/"+FILE_NAME.replace(".set",".pkl"), mode="rb") as g:  
    score_matching_results = pickle.load(g)
    est = score_matching_results["est"]
    Sigma_zero = score_matching_results["Sigma_zero"]

n,d = data_arr.shape #n ; sample size, d : dimension
ls = [] #matrix of test stats
adj_mat = np.zeros((d,d)) #adjancy matrix of final network
alpha = 0.05


for i in range(d):
    tmp_ = []
    for j in range(d):
        if i==j:
            val = 0
        else:
            val = test_for_one_edge(n,d,est,min(i+1,j+1),max(i+1,j+1),Sigma_zero,verbose=False)
            one_minus_alpha = chi2.cdf(x = val, df = 4)            
            if 1-alpha < one_minus_alpha: #reject null hypothesis
                adj_mat[i][j] = 1
        tmp_.append(val)
    ls.append(tmp_)

corr_mat = np.array(ls)
np.fill_diagonal(corr_mat, np.max(corr_mat))
np.save("output/test_stats_matrix",corr_mat)
np.save("output/adjancy_matrix",adj_mat)

correlation.corr_to_corr_map(corr_mat)
correlation.corr_to_corr_map(adj_mat)

# visualize distribution of test stats
plt.hist(corr_mat.flatten().tolist())
plt.vlines(chi2.ppf(0.95, df = 4), 0, 5000, colors='red', linestyle='dashed', linewidth=3)
plt.savefig("output/dist_of_test_stats.png")


print("="*10)
print("Saved correlation matrix and result to output/")
print("="*10)