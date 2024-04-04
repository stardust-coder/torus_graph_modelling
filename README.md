# EEG/ECoG series analysi / 脳神経データ解析

### Files
- torus_graph_model/
    - Torus graph model proposed in [Klein et al.(2020)]()
- utils/
    - preprocessing for neuroscience time series data. 
    - visualization of correlation matrix (and its variant)

### Basic Usage

#### Sample from torus graph
```
python torus_graph_model/sample.py
```

#### Fit Torus graph and view test statistics under null hypothesis (two nodes are independent)
```
# take around 2 hours with 91 dimensional EEG data
python script/analysis_human_eeg.py
```

### Example Usage

1. Place your EEG/ECoG dataset to `PATH_TO_DATA_DIR`
1. Specify `FILE_NAME` to be your target time series data.
1. run `python torus_graph_model/sample.py` and wait.
1. check `output/` for results.

