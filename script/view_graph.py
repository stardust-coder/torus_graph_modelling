import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
color_list = list(mcolors.CSS4_COLORS.keys())
import pickle
import networkx as nx
import pdb
from constant import get_eeg_position, get_eeg_filenames, get_electrode_names
import mne
import sys
sys.path.append(".")

drowsy = [3,6,7,8,10,14,26]
responsive = [2,5,9,13,18,20,22,23,24,27,28,29]
state = ["baseline","mild","moderate","recovery"]

### Specify df (csv)
# df = pd.read_csv("/home/sukeda/torus_graph_modelling/output/estimate_phi_admmpath/beta/result-top100.csv")
def view_ave_cluster():
    for d in drowsy:
        plt.plot(state,df[df["patient_id"]==d]["ave_cluster"].tolist(),color="blue")
    for r in responsive:
        plt.plot(state,df[df["patient_id"]==r]["ave_cluster"].tolist(),color="red")
    plt.ylim(0,1)
    plt.savefig("./output.png")

def view_ave_shortest_path():
    for d in drowsy:
        plt.plot(state,df[df["patient_id"]==d]["ave_shortest_path"].tolist(),color="blue")
    for r in responsive:
        plt.plot(state,df[df["patient_id"]==r]["ave_shortest_path"].tolist(),color="red")
    plt.ylim((1,2))
    plt.savefig("./output.png")

def small_world():
    for d in drowsy:
        plt.plot(state,df[df["patient_id"]==d]["sigma"].tolist(),color="blue")
    for r in responsive:
        plt.plot(state,df[df["patient_id"]==r]["sigma"].tolist(),color="red")
    plt.ylim((0.8,1.2))
    plt.savefig("./output.png")

def modularity():
    for d in drowsy:
        plt.plot(state,df[df["patient_id"]==d]["modularity"].tolist(),color="blue")
    for r in responsive:
        plt.plot(state,df[df["patient_id"]==r]["modularity"].tolist(),color="red")
    plt.ylim((0,0.2))
    plt.savefig("./output.png")



if __name__=="__main__":
    # view_ave_cluster()
    # view_ave_shortest_path()
    # small_world()
    # modularity()

    import networkx as nx

    def charikar_algorithm(G):
        """
        Finds the maximum density subgraph using Charikar's greedy algorithm.

        Args:
            G: A NetworkX graph.

        Returns:
            A subgraph with the highest density.
        """
        subgraph = G.copy()
        max_density = nx.density(G)

        while len(G.nodes()) > 0:
            # Find the node with the lowest degree
            node_to_remove = min(G.nodes(), key=G.degree)

            # Remove the node and its edges from the graph
            G.remove_node(node_to_remove)

            # Calculate the density of the remaining graph
            current_density = nx.density(G)

            # If the density increases, update the subgraph
            if current_density > max_density:
                max_density = current_density
                subgraph = G.copy()

        return subgraph
        

    def draw(patient_id,patient_state_id):
        ###Graph
        ind_list, FILE_NAME_LIST = get_eeg_filenames()
        patient_states = {0:"baseline",1:"mild",2:"moderate",3:"recovery"}
        state_id = ind_list[patient_id][patient_state_id]
        FILE_NAME = f"{FILE_NAME_LIST[patient_id]}{state_id}"
        PATH_TO_DATA_DIR = "../data/Sedation-RestingState/"
        PATH_TO_DATA = PATH_TO_DATA_DIR + FILE_NAME + ".set"
        loaded_eeg = mne.io.read_epochs_eeglab(
                PATH_TO_DATA, verbose=False, montage_units="cm"
            )
        montage = loaded_eeg.get_montage()
        ch_pos = get_eeg_position()
        ch_pos = {k:[v[0],-v[1]] for (k,v) in ch_pos.items()}
        group = "drowsy" if patient_id in drowsy else "responsive"

        filename = f"/home/sukeda/torus_graph_modelling/output/61dim_25000data_rotational_6/{group}/{patient_id}/6_{patient_id}_{patient_state_id}_{patient_states[patient_state_id]}_{state_id}_glasso_model.pkl"
        with open(filename, "rb") as f:
            M = pickle.load(f)

        mapping = {}
        ch_names = get_electrode_names(61)
        for i in range(1,62):
            mapping[i] = ch_names[i-1]
        H = nx.relabel_nodes(M.G, mapping)

        # ###DSP
        # DS = charikar_algorithm(H.copy())
        # pr = nx.pagerank(H)
        # colors = []
        # for name in ch_names:
        #     if name in list(DS.nodes):
        #         colors.append("yellow")
        #     else:
        #         colors.append("white")
        
        # ###Clauset-Newman-Moore greedy modularity maximization
        # c = nx.community.greedy_modularity_communities(H)
        # colors = []
        # for name in ch_names:
        #     for k in range(len(c)):
        #         if name in c[k]:
        #             colors.append(color_list[k])
        
        ### Modularity maximization by Louvain
        c = nx.community.louvain_communities(H, seed=123)
        colors = []
        for name in ch_names:
            for k in range(len(c)):
                if name in c[k]:
                    colors.append(color_list[k])

        ###Degrees
        degrees = np.array([d for n, d in H.degree()])
        sizes = degrees**2 
        cmap = plt.get_cmap("Pastel1")
        node_color = [cmap(i) for i in range(H.number_of_nodes())]
        labels = {}
        for i in range(H.number_of_nodes()):
            labels[ch_names[i]] = ch_names[i] + "\ndeg:" + str(degrees[i])


        ###Plot
        plt.figure(figsize=(10,10))
        options = {"edgecolors": "tab:gray", "alpha": 0.9}
        nx.draw_networkx_nodes(H, ch_pos, node_size=sizes, node_color=colors, **options)
        nx.draw_networkx_edges(H, ch_pos, width=1.0, alpha=0.5,edge_color="tab:red")
        nx.draw_networkx_labels(H, ch_pos,labels=labels, font_size=9, font_color="black")

        plt.title(f"{H.number_of_edges()} edges")
        plt.tight_layout()
        plt.axis("off")
        plt.show()
        plt.savefig(f"{patient_id}-{patient_state_id}-degree.png")
        

    draw(7,0)
    draw(7,1)
    draw(7,2)
    draw(7,3)
    
    
