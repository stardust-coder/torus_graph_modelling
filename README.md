# Regularized Score Matching of Torus Graph Model for EEG Phase-connectivity Analysis

* Previous study (Klein et al.) : Fit torus graph to multivariate circular data and view test statistics under null hypothesis that two nodes are independent. 

* Ours : Regularized score matching of the torus graph model to obtain sparse solutions.

* Note: These models do not handle time series information.

## Files
- model/
    - model (old)
    - torus_graph (new) : class
    - torus_graph_cupy (new) : accelerated with cupy, only in GPU environment
- output: the results will be saved.
- utils/
    - utils: preprocessing for neuroscience time series data. 
    - visualize: visualization of correlation matrix (and its variant)
    - simulation : for simulation studies
- script
    - bias_correction.py : to check SMIC is correcting the inherit bias. long computation.
    - main : to run inference. It takes around 15min for naive estimation and additional 15~30min for Group LASSO estimation.
    - constant : file names and so on.
    - run_paper_experiment : reproduction of results in our paper
    - view*.py : to observe estimated graph structures and properties

## Basic Usage
```python
import sys
sys.path.append(".")
from model.torus_graph import Torus_Graph
from utils.simulation import sample_from_torus_graph, star_shaped_sample
import numpy as np       

data_arr = star_shaped_sample(1000) # 5 dimensional sample from a torus graph
M = Torus_Graph(5)
M.estimate(data_arr,mode="naive")
# M.estimate(data_arr,mode="lasso") #If you want to use lasso, run it after naive estimation, otherwise it fails.
M.plot(weight=True)
```

With Cupy(GPU) acceleration
```python
import sys
sys.path.append(".")
from model.torus_graph_cupy import Torus_Graph
from utils.simulation import sample_from_torus_graph, star_shaped_sample
import numpy as np       

data_arr = star_shaped_sample(1000) # 5 dimensional sample from a torus graph
M = Torus_Graph(5)
M.estimate(data_arr,mode="naive")
M.plot(weight=True)
```

## Mode
- lasso : regularization on full parameters equally
- glasso : regularization on d nodes and on d*(d-1)/2 edges by group (Group LASSO)

## Run experiment in our paper (61 dimensional EEG phase analysis)
```
python script/main.py
```

## Run comparison SMCV vs SMIC
CV1() runs 5 dimensional Group LASSO.  (100 candidates)
CV2() runs 3 dimensional naive estimation. ($2^3$ candidates)

```
python script/crossvalidation.py
```

## Visualize reconstucted graph
Set path to load.
```script/view_graph.py
...
filename = "PATH TO YOUR PKL FILE"
...
```
Then run it to output images with colored edges and nodes. The size of nodes reflects the degree. The color of nodes represents community structures.
```
python script/view_graph.py
```

## Acknowledgement
This repository is partly a reimplementation of [Klein et al, 2020](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-14/issue-2/Torus-graphs-for-multivariate-phase-coupling-analysis/10.1214/19-AOAS1300.full).

We thank the developers of the EEG dataset we used [Chennu et al., 2016](https://www.repository.cam.ac.uk/items/b7817912-50b5-423b-882e-978fb39a49df).


## How to cite
```
coming soon...
```
