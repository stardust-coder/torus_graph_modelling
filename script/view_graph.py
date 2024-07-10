import pandas as pd
import matplotlib.pyplot as plt
import pdb

df = pd.read_csv("/home/sukeda/torus_graph_modelling/output/estimate_phi_admmpath/beta/result-top100.csv")

drowsy = [3,6,7,8,10,14,26]
responsive = [2,5,9,13,18,20,22,23,24,27,28,29]
state = ["baseline","mild","moderate","recovery"]

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
    modularity()
