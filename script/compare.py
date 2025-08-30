#Compare Sparse torus graph (ours) and Non-sparse torus graph (Klein et al.) on simulation dataset. 61dim.

import sys
sys.path.append(".")
import numpy as np
from model.torus_graph_model import Torus_Graph_Model
from utils.gibbs import gibbs_sampler
import networkx as nx
import matplotlib.pyplot as plt

def plot_binary(M, edges, filename):
    # nodes
    nx.draw_networkx_nodes(M.G, M.pos, node_size=700)

    # edges
    nx.draw_networkx_edges(M.G, M.pos, edgelist=edges, width=2, edge_color="r")

    # node labels
    nx.draw_networkx_labels(M.G, M.pos, font_size=20, font_family="sans-serif", font_color="white")

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(filename)
    plt.close()

def plot_weighted(M, threshold, filename):
    # nodes
    nx.draw_networkx_nodes(M.G, M.pos, node_size=700)

    # edges
    elarge = [(u, v) for (u, v, d) in M.G.edges(data=True) if d["weight"] > threshold]
    esmall = [(u, v) for (u, v, d) in M.G.edges(data=True) if d["weight"] <= threshold]
    nx.draw_networkx_edges(M.G, M.pos, edgelist=elarge, width=2, edge_color="r")
    nx.draw_networkx_edges(
        M.G, M.pos, edgelist=esmall, width=2, alpha=0.5, edge_color="b", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(M.G, M.pos, font_size=20, font_family="sans-serif", font_color="white")
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(filename)
    plt.close()

data_arr, true_edge = gibbs_sampler(N=10000)

M_orig = Torus_Graph_Model(19)
plot_binary(M_orig, true_edge, "output/truth.png")

M_orig.estimate(data_arr,mode="naive",img_path=f"output/tmp_orig_naive.png")
M_orig.graph_property(abbr=False)
plot_weighted(M_orig, 0.1, "output/thresh0.1.png")
plot_weighted(M_orig, 0.3, "output/thresh0.3.png")
plot_weighted(M_orig, 0.5, "output/thresh0.5.png")


M = Torus_Graph_Model(19)
M.estimate(data_arr,mode="naive",img_path=f"output/tmp_naive.png")
M.lambda_list = [0] + np.logspace(-3,0.5, num=99).tolist()
M.glasso_weight = [0 for _ in range(2*M.d)] + [1 for _ in range(2*M.d*M.d-2*M.d)]
for _ in range(1):
    M.estimate(data_arr,mode="glasso",img_path=f"output/tmp_glasso.png")
    opt_index = M.smic.index(min(M.smic))
    if opt_index == 0 or opt_index == len(M.lambda_list)-1:
        break
    M.lambda_list = np.linspace(M.lambda_list[opt_index-1],M.lambda_list[opt_index+1],10)

plot_binary(M, M.G.edges, "output/ours.png")


print("=" * 30)
print("Estimated edges")
print(M.G.edges) #推定された辺
print("Estimated degree distribution")
print(M.G.degree())#次数の分布
print("=" * 30)
M.graph_property(abbr=False)

print(M.smic)
print(M.reg_path)