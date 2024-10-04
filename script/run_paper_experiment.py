import sys
sys.path.append(".")
use_cupy = False
if use_cupy:
    from model.full_model_cupy import Torus_Graph_Model
    from model.rotational_model_cupy import Rotational_Model
else:
    from model.full_model import Torus_Graph_Model
    from model.rotational_model import Rotational_Model


from utils import utils, correlation
from utils.simulation import sample_from_torus_graph, star_shaped_sample, star_shaped_rotational_sample, bagraph_sample
from observe import draw_heatmap
from data.dataloader import chennu, chennu_with_pos
from constant import get_eeg_filenames, get_electrode_names

import numpy as np
import matplotlib.pyplot as plt
import pickle

def save(model,output_path):
    model.Gamma_hat =  None
    model.H_hat = None
    model.D_list = None
    model.H_list = None
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

def run(exp_id,patient_id,patient_state_id):
    ind_list, FILE_NAME_LIST = get_eeg_filenames()
    patient_states = {0:"baseline",1:"mild",2:"moderate",3:"recovery"}
    state_id = ind_list[patient_id][patient_state_id]
    out_id = f"{exp_id}_{patient_id}_{patient_state_id}_{patient_states[patient_state_id]}_{state_id}"

    data_arr = chennu(patient_id=patient_id,state_id=state_id,dim=61)
    correlation.data_to_corr_map(data_arr,utils.PLV,f"output/{out_id}_PLV.png")
    plt.clf()
    correlation.data_to_corr_map(data_arr,utils.PLI,f"output/{out_id}_PLI.png")
    plt.clf()

    # M = Torus_Graph_Model(61)
    M = Rotational_Model(61)

    ###Naive
    M.estimate(data_arr,mode="naive")
    # save(M,output_path=f"output/{out_id}_naive.pkl")
    draw_heatmap(M,output_img_path=f"output/{out_id}_naive_heatmap.png")
    M.graph_property(abbr=True)

    ###GroupLASSO with repeat lambda
    M.glasso_weight = [0 for _ in range(2*M.d)] + [1 for _ in range(2*M.d*M.d-2*M.d)]
    for _ in range(6):
        M.estimate(data_arr,mode="glasso",img_path=f"output/{out_id}_glasso.png")
        opt_index = M.smic.index(min(M.smic))
        if opt_index == 0 or opt_index == 9:
            break
        M.lambda_list = np.linspace(M.lambda_list[opt_index-1],M.lambda_list[opt_index+1],10)
    draw_heatmap(M,output_img_path=f"output/{out_id}_glasso_heatmap.png")
    M.graph_property(abbr=True)
    save(M,output_path=f"output/{out_id}_glasso_model.pkl")

    print("=" * 30)
    print(M.G.edges) #推定された辺
    print(M.G.degree())#次数の分布
    print("=" * 30)

if __name__ == "__main__":
    import argparse    # 1. argparseをインポート
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--experiment') 
    parser.add_argument('-p', '--patient') 
    parser.add_argument('-s', '--state') 
    args = parser.parse_args()   

    # print(args.experiment, args.patient, args.state)

    run(int(args.experiment), int(args.patient), int(args.state))