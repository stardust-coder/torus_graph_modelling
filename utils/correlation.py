from utils import *
import matplotlib.pyplot as plt
import numpy as np

def corr_to_corr_map(mat):
    plt.imshow(mat,cmap="jet")
    plt.colorbar()
    plt.show()


def data_to_corr_map(data, metric):
    '''
    data : (t,d)
    metric : function that takes two values and return one float

    Output :
        corr_list : d x d matrix and image
    '''
    d = data.shape[1]
    corr_list = []
    for i in range(d):
        tmp_ = []
        for j in range(d):
            x = data[:,i]
            y = data[:,j]
            tmp_.append(metric(x,y))
        corr_list.append(tmp_)
    corr_list = np.array(corr_list)
    plt.imshow(corr_list,cmap="jet")
    plt.colorbar()
    plt.show()
