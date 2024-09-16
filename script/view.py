import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("/home/sukeda/torus_graph_modelling/output/61dim_25000data_rotational/result.tsv", sep="\t")
df = df.fillna("(None,None,None,None,None)")
print(df)

drowsy = [3,6,7,8,10,14,26]
responsive = [2,5,9,13,18,20,22,23,24,27,28,29]
states = ["baseline","mild","moderate","recovery"]
measures = ["#edge", "modularity", "average clustering", "average shortest path length","small-world coefficient"]
measure = 4


plt.figure(figsize=(15,10))
plt.xticks([0,1,2,3],states)
plt.xlabel("Patient state")
plt.ylabel(measures[measure])


for drowsy_patient in drowsy:
    df_ind = df[df["patient_ID"]==drowsy_patient]
    data = [eval(df_ind[s].item())[measure] for s in states]
    plt.plot(data, c="b", label=f"{drowsy_patient}")

for responsive_patient in responsive:
    df_ind = df[df["patient_ID"]==responsive_patient]
    data = [eval(df_ind[s].item())[measure] for s in states]
    plt.plot(data, c="r", label=f"{responsive_patient}")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
plt.savefig("seni.png")
