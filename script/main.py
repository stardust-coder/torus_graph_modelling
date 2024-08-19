import sys
sys.path.append(".")
# from model.torus_graph import Torus_Graph
from model.torus_graph_cupy import Torus_Graph
import numpy as np
from utils.simulation import sample_from_torus_graph, star_shaped_sample
from data.dataloader import chennu, chennu_with_pos
from constant import get_eeg_filenames, get_electrode_names
import matplotlib.pyplot as plt

#simulation
# errors = []
# for _ in range(30):
#     data_arr = star_shaped_sample(10000)
#     M1 = Torus_Graph(5)
#     M2 = Torus_Graph(5)
#     M1.estimate(data_arr,mode="naive")
#     M2.estimate_by_edge(data_arr,mode="naive")
#     error = np.linalg.norm(M1.param-M2.param)
#     errors.append(error)
# import pdb; pdb.set_trace()

# data_arr = star_shaped_sample(10000)
# M = Torus_Graph(5)
# M.estimate(data_arr,mode="naive")
# print(M.param.T.tolist()[0])
# M.estimate(data_arr,mode="lasso")
# print(M.param.T.tolist()[0])
# M.estimate(data_arr,mode="glasso")
# print(M.param.T.tolist()[0])
# M.glasso_weight = [0 for _ in range(10)] + [1 for _ in range(40)]
# M.estimate(data_arr,mode="glasso")
# print(M.param.T.tolist()[0])
# import pdb; pdb.set_trace()

#main
# data_arr = star_shaped_sample(1000)

exp_id = 0
patient_id = 2
patient_state_id = 0

ind_list, FILE_NAME_LIST = get_eeg_filenames()
patient_states = {0:"baseline",1:"mild",2:"moderate",3:"recovery"}
state_id = ind_list[patient_id][patient_state_id]
data_arr = chennu(patient_id=patient_id,state_id=state_id,dim=61)

M = Torus_Graph(61)
M.estimate(data_arr,mode="naive")
M.estimate(data_arr,mode="lasso",img_path=f"{exp_id}_{patient_id}_{patient_state_id}_{patient_states[patient_state_id]}_{state_id}_lasso.png")
M.glasso_weight = [0 for _ in range(2*M.d)] + [1 for _ in range(2*M.d*M.d-2*M.d)]
M.estimate(data_arr,mode="glasso",img_path=f"{exp_id}_{patient_id}_{state_id}_glasso.png")
# M.estimate_by_edge(data_arr,mode="lasso")
# import pdb; pdb.set_trace()