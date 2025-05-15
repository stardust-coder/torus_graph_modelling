import sys
sys.path.append(".")

use_cupy = False
if use_cupy:
    from model.full_model_cupy import Torus_Graph_Model
    from model.rotational_model_cupy import Rotational_Model
else:
    from model.torus_graph_model import Torus_Graph_Model
    from model.rotational_model import Rotational_Model
    from model.cylinder_model import Cylinder_Graph_Model

from utils.simulation import sample_from_torus_graph, star_shaped_sample, star_shaped_rotational_sample, bagraph_sample
from utils.gibbs import gibbs_sampler
from utils.utils import draw_heatmap
from data.dataloader import chennu, chennu_with_pos
from constant import get_eeg_filenames, get_electrode_names

import numpy as np
import matplotlib.pyplot as plt
import pickle

def save(model,output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)

def AV(data):
    ave_reference = np.mean(data,axis=1)
    data = data - ave_reference.reshape(-1,1)
    return data

###naive vs conditional
# errors = []
# for _ in range(30):
#     data_arr = star_shaped_sample(10000)
#     M1 = Torus_Graph_Model(5)
#     M2 = Torus_Graph_Model(5)
#     M1.estimate(data_arr,mode="naive")
#     M2.estimate_by_edge(data_arr,mode="naive")
#     error = np.linalg.norm(M1.param-M2.param)
#     errors.append(error)
# import pdb; pdb.set_trace()

###simulation
# data_arr, true_edges = gibbs_sampler(25000)
# data_arr = star_shaped_sample(1000)
# data_arr = star_shaped_rotational_sample(1000)

# # M = Rotational_Model(data_arr.shape[1])
# M = Torus_Graph_Model(data_arr.shape[1])

# M.estimate(data_arr,mode="naive")
# print(M.param.T.tolist()[0])
# # # M.estimate(data_arr,mode="lasso")
# # # print(M.param.T.tolist()[0])
# # # M.estimate(data_arr,mode="glasso")
# # # print(M.param.T.tolist()[0])
# M.glasso_weight = [0 for _ in range(2*M.d)] + [1 for _ in range(M.model_d-2*M.d)]
# M.estimate(data_arr,mode="glasso")
# print(M.param.T.tolist()[0])


#Main codes
exp_id = 0
patient_id = 2
patient_state_id = 0
ind_list, FILE_NAME_LIST = get_eeg_filenames()
patient_states = {0:"baseline",1:"mild",2:"moderate",3:"recovery"}
state_id = ind_list[patient_id][patient_state_id]
out_id = f"{exp_id}_{patient_id}_{patient_state_id}_{patient_states[patient_state_id]}_{state_id}"

phase_data_arr, amp_data_arr = chennu(patient_id=patient_id,state_id=state_id,dim=61) #choose dim from {5,19,61,91}
# data_arr = phase_data_arr
data_arr = [phase_data_arr,amp_data_arr]


# data_arr = AV(data_arr) #minus Average 
# correlation.data_to_corr_map(data_arr,utils.PLV,f"output/{out_id}_PLV.png")
# plt.clf()
# correlation.data_to_corr_map(data_arr,utils.PLI,f"output/{out_id}_PLI.png")
# plt.clf()

M = Torus_Graph_Model(data_arr.shape[1])
# M = Rotational_Model(data_arr.shape[1])

###Naive
M.estimate(data_arr,mode="naive")
save(M,output_path=f"output/{out_id}_naive.pkl")
draw_heatmap(M,output_img_path=f"output/{out_id}_naive_heatmap.png")
M.graph_property(abbr=True)

###GroupLASSO with repeat lambda
M.lambda_list = [0] + np.logspace(-3,0.5, num=99).tolist()
M.glasso_weight = [0 for _ in range(2*M.d)] + [1 for _ in range(2*M.d*M.d-2*M.d)]
for _ in range(1):
    M.estimate(data_arr,mode="glasso",img_path=f"output/{out_id}_glasso.png")
    opt_index = M.smic.index(min(M.smic))
    M.lambda_list = np.linspace(M.lambda_list[opt_index-1],M.lambda_list[opt_index+1],10)
# draw_heatmap(M,output_img_path=f"output/{out_id}_glasso_heatmap.png")
# M.graph_property(abbr=True)

###GroupLASSO
# M.glasso_weight = [0 for _ in range(2*M.d)] + [1 for _ in range(2*M.d*M.d-2*M.d)]
# M.estimate(data_arr,mode="glasso",img_path=f"output/{out_id}_glasso.png")
# # save(M,output_path=f"output/{out_id}_glasso.pkl")
# draw_heatmap(M,output_img_path=f"output/{out_id}_glasso_heatmap.png")
# M.graph_property(abbr=True)

###LASSO
# M.estimate(data_arr,mode="lasso",img_path=f"output/{out_id}_lasso.png")
# save(M,output_path=f"output/{out_id}_lasso.pkl")
# draw_heatmap(M,output_img_path=f"output/{out_id}_lasso_heatmap.png")
# M.graph_property(abbr=False)