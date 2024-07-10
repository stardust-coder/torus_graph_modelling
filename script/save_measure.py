import os
import sys
import pdb
import pickle
import networkx as nx
import numpy as np
import glob
import pandas as pd
import community
from tqdm import tqdm

def print_graph_features(edges):
    G = nx.Graph()
    G.add_nodes_from(list(range(1,19+1)))
    G.add_edges_from(edges)
    print(G)
    print("Average clustering coefficient = ", nx.average_clustering(G))
    try:
        ave_spl = nx.average_shortest_path_length(G)
    except:
        ave_spl = None

    partition = community.best_partition(G)
    modularity = community.modularity(partition,G)
    ave_clus = nx.average_clustering(G)
    G = G.subgraph(max(nx.connected_components(G),key=len)).copy() #連結でないとsigmaやomegaはerrorになる...
    s_ = nx.sigma(G)
    o_ = nx.omega(G)
    res_ = [ave_clus,ave_spl,s_,o_,modularity,len(edges)]
    print(res_)
    return res_

if __name__ == "__main__":
    df = []
    query = 100 #上位50/100本の辺を見る

    # for exp in [6,7,9,14,18,20]:
    for exp in tqdm([2,3,5,6,7,8,9,10,13,14,18,20,22,23,24,26,27,28,29]):
        original_stdout = sys.stdout

        # FOLDER = f"estimate_phi_admm_0.5/{exp}"
        FOLDER = f"estimate_phi_admmpath/beta/{exp}"

        with open(f"output/{FOLDER}/measure_top{query}.txt", 'w') as f2:
            f2.write("")

        phase = 0
        # for f in glob.glob(f"pickles/{FOLDER}/*.pkl"):
        # for f in glob.glob(f"pickles/{FOLDER}/*best.pkl"):
        for f in glob.glob(f"pickles/{FOLDER}/*path.pkl"):
            phase += 1 #1:baseline ~ 4:recover
            FILE = os.path.split(f)[1]
            # FILE = "03-2010-anest 20100211 142.003_dict.pkl"
            with open(f"pickles/{FOLDER}/{FILE}", mode="rb") as f1:
                est = pickle.load(f1)  # load best result

            
            
        
            with open(f"output/{FOLDER}/measure_top{query}.txt", 'a') as f3:
                sys.stdout = f3

                #####ADMM PATHの結果を利用する場合、ここから#############################################
                def count_edges(est):
                    num_of_edge = 0
                    for item in est.items():
                        if item[0][0] == 0:
                            pass
                        else:
                            thresh = 1e-5
                            if np.linalg.norm(item[1]) >= thresh:
                                num_of_edge += 1
                    return num_of_edge
                list_of_num_of_edge = [abs(count_edges(e)-query) for e in est]
                queried_index = list_of_num_of_edge.index(min(list_of_num_of_edge))
                est = est[queried_index]
                print(f"Index closest to {query} edges: {queried_index}")
                #####ここまで########################################
                
                edges = []
                for item in est.items():
                    if item[0][0] == 0:
                        pass
                    else:
                        thresh = 1e-5
                        if np.linalg.norm(item[1]) >= thresh:
                            edges.append(item[0])
                print(FILE)
                res = print_graph_features(edges)
                print("")
            
            res = [exp, phase, FILE] + res
            df.append(res)
    df = pd.DataFrame(df, columns=["patient_id","phase","file","ave_cluster","ave_shortest_path","sigma","omega","modularity","edge"])

    sys.stdout = original_stdout
    
    df.to_csv("output/result.csv",index=False)