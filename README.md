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
    - main : 
    - exp : experiments for our paper
    - constant : file names and so on.

## Basic Usage (New)
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
- glasso : regularization on d nodes and on d*(d-1)/2 edges by group

## Run experiment in our paper (61 dimensional EEG phase analysis)
```
python script/main.py
```

## Run comparison SMCV vs SMIC
```
python script/crossvalidation.py
```

## Acknowledgement
This repository is partly a reimplementation of [Klein et al, 2020](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-14/issue-2/Torus-graphs-for-multivariate-phase-coupling-analysis/10.1214/19-AOAS1300.full).

We thank the developers of the EEG dataset [Chennu et al., 2016](https://www.repository.cam.ac.uk/items/b7817912-50b5-423b-882e-978fb39a49df).


## TODO 
- implement dataloader for any data


## How to cite
```
coming soon...
```

---

## Basic Usage (Old)

### Sample from torus graph
```
python torus_graph_model/sample.py
```

###　Use your time series data
1. Place your time series data in CSV format.
2. Run
```
python scripts/inference.py <path_to_your_csv> #if it is a raw signal
python scripts/inference.py <path_to_your_csv> --phase #if it is already a circular data series
```

### Use human EEG data from [Chennu et al., 2016](https://www.repository.cam.ac.uk/items/b7817912-50b5-423b-882e-978fb39a49df)
```
python script/score_matching.py #naive matrix inversion or conditional models
python script/score_matching_admmpath.py #ADMM with regularization path and SMIC minimization


# confirmation using simulation data
python script/simulation.py

```

#### About this data
1. baseline
2. mild
3. moderate
4. recover

#### Optimization Method
Score matching estimator using
- without regularization (full model)
- without regularization (conditional model)
- $l_1$ regularization using gradient descent (not recommended)
- LASSO with ADMM
- LASSO with LARS and SMIC

### Example Usage

1. Place your EEG/ECoG dataset to `PATH_TO_DATA_DIR`
1. Specify `FILE_NAME` to be your target time series data.
1. run `python torus_graph_model/sample.py` and wait.
1. check `output/` for results.
