import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# color_list = list(mcolors.CSS4_COLORS.keys())
color_list = [matplotlib.cm.tab20(i) for i in range(20)]

import itertools
import pickle
import networkx as nx
import sys
sys.path.append(".")


#グラフ
filename = "output/MarmosetECoG/Ji_β/Marmoset_ECoG_3_glasso_model.pkl"
with open(filename, "rb") as f:
    M = pickle.load(f)

### Modularity maximization by Louvain
community = nx.community.louvain_communities(M.G, seed=123)

def node_to_color(node):
    tmp = 0
    while True:
        if node in community[tmp]:
            break
        else:
            tmp += 1
    return color_list[tmp]


#可視化
marmoset_name = "Ji"
assert marmoset_name in filename
electrodes = scipy.io.loadmat(f'../data/riken-auditory-ECoG/Electrodes{marmoset_name}/Electrodes{marmoset_name}.mat')
print("Num of electrodes:",len(electrodes["X"]))

plt.figure(figsize=(15,15))

###グラフを描画
count = 0
for i,j in itertools.combinations(list(range(96)),2):
    if (i+1,j+1) in M.G.edges:
        plt.plot([electrodes["X"][i].item(), electrodes["X"][j].item()], [electrodes["Y"][i].item(), electrodes["Y"][j].item()], color="gray", lw=0.5)
        count += 1

#脳の描画
plt.imshow(electrodes["LINE"])
#電極の分布
#plt.scatter(electrodes["X"],electrodes["Y"])
#電極に番号をアノテーション
coord = []
for (i,j,k) in zip(electrodes["X"],electrodes["Y"],list(range(1,97))):
    plt.plot(i[0],j[0],marker="o",color=node_to_color(k))
    if ((i[0],j[0])) in coord:
        plt.annotate(f"&{k}", xy=(i[0]+7, j[0]))
    else:
        plt.annotate(k, xy=(i[0]+3, j[0]))
    
    coord.append((i[0],j[0]))


plt.savefig(f"{marmoset_name}_graph.png")

print(len(M.G.edges))
print("num of edges:", count)