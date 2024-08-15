import sys
sys.path.append(".")
from model.torus_graph import Torus_Graph
from utils.simulation import sample_from_torus_graph, star_shaped_sample
import numpy as np       

data_arr = star_shaped_sample(1000)
M = Torus_Graph(5)
M.estimate(data_arr,mode="naive")
# M.estimate(data_arr,mode="lasso")
M.plot(weight=True)