# EEG,ECoG phase series analysis using Torus Graph / 脳神経データ解析

Fit torus graph to multivariate circular data and view test statistics under null hypothesis that two nodes are independent. 

Note: This model does not handle time series information.


## Files
- torus_graph_model/
    - Torus graph model
- utils/
    - preprocessing for neuroscience time series data. 
    - visualization of correlation matrix (and its variant)

## Basic Usage

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

### Acknowledgement
This repository is partly a reimplementation of [Klein et al, 2020](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-14/issue-2/Torus-graphs-for-multivariate-phase-coupling-analysis/10.1214/19-AOAS1300.full)