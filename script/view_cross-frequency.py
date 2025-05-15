import matplotlib.pyplot as plt
import pandas as pd
df_alpha = pd.read_csv("/home/sukeda/torus_graph_modelling/output/61dim_25000data_rotational_5（α）/result.tsv", sep="\t")
df_beta = pd.read_csv("/home/sukeda/torus_graph_modelling/output/61dim_25000data_rotational_7（β）/result.tsv", sep="\t")
df_gamma = pd.read_csv("/home/sukeda/torus_graph_modelling/output/61dim_25000data_rotational_8（γ）/result.tsv", sep="\t")


drowsy = [3,6,7,8,10,14,26]
responsive = [2,5,9,13,18,20,22,23,24,27,28,29]
states = ["baseline","mild","moderate","recovery"]
measures = ["#edge", "modularity", "average clustering", "average shortest path length","small-world coefficient","pseudo small-world"]
measure = 1


plt.figure(figsize=(10,10))
plt.xlabel(f"{measures[measure]} of alpha band")
plt.ylabel(f"{measures[measure]} of gamma band")


for drowsy_patient in drowsy:
    df_alpha_ind = df_alpha[df_alpha["patient_ID"]==drowsy_patient]
    df_beta_ind = df_beta[df_beta["patient_ID"]==drowsy_patient]
    df_gamma_ind = df_gamma[df_gamma["patient_ID"]==drowsy_patient]

    if measure in [0,1,2,3]:
        data_alpha = [eval(df_alpha_ind[s].item())[measure] for s in states]
        data_beta = [eval(df_beta_ind[s].item())[measure] for s in states]
        data_gamma = [eval(df_gamma_ind[s].item())[measure] for s in states]

    elif measure == 5:
        data_alpha = [eval(df_alpha_ind[s].item())[2]/eval(df_alpha_ind[s].item())[3] for s in states]
        data_beta = [eval(df_beta_ind[s].item())[2]/eval(df_beta_ind[s].item())[3] for s in states]
        data_gamma = [eval(df_gamma_ind[s].item())[2]/eval(df_gamma_ind[s].item())[3] for s in states]


    # plt.scatter(data_alpha,data_beta, c="b", label=f"{drowsy_patient}")
    plt.scatter(data_alpha,data_gamma, c="b", label=f"{drowsy_patient}")
    # plt.scatter(data_beta,data_gamma, c="b", label=f"{drowsy_patient}")


for responsive_patient in responsive:
    df_alpha_ind = df_alpha[df_alpha["patient_ID"]==responsive_patient]
    df_beta_ind = df_beta[df_beta["patient_ID"]==responsive_patient]
    df_gamma_ind = df_gamma[df_gamma["patient_ID"]==responsive_patient]

    if measure in [0,1,2,3]:
        data_alpha = [eval(df_alpha_ind[s].item())[measure] for s in states]
        data_beta = [eval(df_beta_ind[s].item())[measure] for s in states]
        data_gamma = [eval(df_gamma_ind[s].item())[measure] for s in states]

    elif measure == 5:
        data_alpha = [eval(df_alpha_ind[s].item())[2]/eval(df_alpha_ind[s].item())[3] for s in states]
        data_beta = [eval(df_beta_ind[s].item())[2]/eval(df_beta_ind[s].item())[3] for s in states]
        data_gamma = [eval(df_gamma_ind[s].item())[2]/eval(df_gamma_ind[s].item())[3] for s in states]

    
    # plt.scatter(data_alpha,data_beta, c="r", label=f"{responsive_patient}")
    plt.scatter(data_alpha,data_gamma, c="r", label=f"{responsive_patient}")
    # plt.scatter(data_beta,data_gamma, c="r", label=f"{responsive_patient}")


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
plt.savefig("view-cross-freq.png")