import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
import itertools

thresh = 3
### load from pickle
for id in ["005","010","023","028"]:
    FILE_NAME= f"29-2010-anest 20100921 142.{id}"

    with open(f"./pickles/18/"+FILE_NAME+".pkl", mode="rb") as g:  
        score_matching_results = pickle.load(g)
        est = score_matching_results["est"]

    n,d = 2500,91 #n ; sample size, d : dimension
    ls = [] #matrix of test stats
    adj_mat = np.zeros((d,d)) #adjancy matrix of final network
    l = [i for i in range(1, d+1)]
    ind_list = [v for v in itertools.combinations(l, 2)]

    for i in range(d):
        tmp_ = []
        for j in range(d):
            if i==j:
                val = 0
            else:
                ind_ = ind_list.index((min(i+1,j+1), max(i+1,j+1)))
                phi_est = est[2*d+4*ind_:2*d+4*(ind_+1)]
                val = np.linalg.norm(phi_est)
            tmp_.append(val)
            if val > thresh:
                adj_mat[i][j]=1
        ls.append(tmp_)

    corr_mat = np.array(ls)
    print(corr_mat.max(),corr_mat.min())
    im = plt.imshow(corr_mat,cmap="jet")
    cb = plt.colorbar(im)
    plt.savefig(f"corr_matrix_thresh={thresh}_{id}.png")
    cb.remove() 

    im = plt.imshow(adj_mat,cmap="jet")
    cb = plt.colorbar(im)
    cb.remove() 
    plt.savefig(f"adjancy_matrix_thresh={thresh}_{id}.png")