import sys
sys.path.append(".")
from model.torus_graph import Torus_Graph_Model
import pickle
import matplotlib.pyplot as plt
import itertools
import pdb
import numpy as np
from constant import get_eeg_filenames, get_electrode_names

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

if __name__=="__main__":
    exp_id = 1
    patient_id = 3
    # patient_state_id = 0
    for patient_state_id in [0,1,2,3]:
        ind_list, FILE_NAME_LIST = get_eeg_filenames()
        patient_states = {0:"baseline",1:"mild",2:"moderate",3:"recovery"}
        state_id = ind_list[patient_id][patient_state_id]
        out_id = f"{exp_id}_{patient_id}_{patient_state_id}_{patient_states[patient_state_id]}_{state_id}"
        # filename = f"output/{out_id}_naive"
        filename = f"output/{out_id}_glasso"

        with open(f"{filename}.pkl","rb") as f:
            M = pickle.load(f)

        draw_heatmap(M,filename+"_heatmap.png")

        M.graph_property(abbr=True)

    pdb.set_trace()

