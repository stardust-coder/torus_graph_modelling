import sys
sys.path.append(".")
from model.torus_graph import Torus_Graph
from utils.simulation import sample_from_torus_graph, star_shaped_sample
import numpy as np       
from data.dataloader import chennu, chennu_with_pos
from constant import get_eeg_filenames, get_electrode_names
# data_arr = star_shaped_sample(1000)
ind_list, FILE_NAME_LIST = get_eeg_filenames()

exp_id = 0
patient_id = 6
state_id = ind_list[patient_id][0]
# data_arr = chennu(patient_id=patient_id,state_id=state_id,dim=61)
data_arr, pos = chennu_with_pos(patient_id=patient_id,state_id=state_id,dim=19)
import pdb; pdb.set_trace()


M = Torus_Graph(61)
M.estimate(data_arr,mode="naive")
import pdb; pdb.set_trace()

M.estimate(data_arr,mode="lasso")
import pdb; pdb.set_trace()