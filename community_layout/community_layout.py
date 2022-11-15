import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx.algorithms.community as community
from collections import Counter
from tqdm import tqdm


def load_original():
    filename = "../example_data/musae_facebook_edges.csv"

    social_data = pd.read_csv(filename, delimiter=",", skiprows=0, dtype = str)
    G = nx.from_pandas_edgelist(social_data, "id_1", "id_2", create_using=nx.Graph)

    return G

def communities(G):
    partition = community.louvain_communities(G, resolution=4)
    print(partition)
    return partition

def make_meta_graph(G):
    community_to_node = {i:p for i,p in enumerate(communities(G))}
    partition = {}
    for comm in community_to_node:
        nodes = community_to_node[comm]
        for n in nodes:
            partition[n] = comm


    community_unique = set([k for k in community_to_node.keys()])
    subgraphs = []

    for c in community_unique:
        subgraphs.append(nx.subgraph(G, community_to_node[c]))

    G_edgelist = [[e1, e2] for (e1, e2) in nx.edges(G)]

    community_edgelist = []

    for e in G_edgelist:
        comm1 = partition[e[0]]
        comm2 = partition[e[1]]

        community_edgelist.append((comm1, comm2))

    unique_comm_edges = list(set(community_edgelist))
    out_edges = []
    for e in unique_comm_edges:
        if (e[1], e[0]) not in out_edges and e[0] != e[1]:
            out_edges.append(e)
    unique_comm_edges = out_edges


    edge_count = [community_edgelist.count(e) + community_edgelist.count([e[1], e[0]]) for e in unique_comm_edges]

    full_description = [(*list(unique_comm_edges)[i], edge_count[i]) for i in range(len(edge_count))]
    print(repr(full_description))
    metaG = nx.Graph()


    metaG.add_weighted_edges_from(full_description)

    # print(metaG.edges[0])
    subgraphs = {i:g for i, g in enumerate(subgraphs)}

    return metaG, subgraphs

def meta_position(G):
    fig, ax = plt.subplots(figsize=(12,12))
    print(f"Metagraph is a {G}")
    weights = np.array(list(nx.get_edge_attributes(G,'weight').values()))
    weights = weights / np.max(weights)
    # print(f"Weights: {weights}")

    # pos = nx.kamada_kawai_layout(G, weight="weight")
    pos = nx.spring_layout(G, weight="weight", k = 100, iterations=1000)
    # pos = nx.nx_agraph.graphviz_layout(G, prog = "neato")
    nx.draw_networkx_edges(G, pos, node_size=20, ax = ax, width=5*weights  + 0.1)
    nx.draw_networkx_nodes(G, pos, node_size=20, ax=ax)


    plt.savefig("metagraph.png")
    plt.close()
    return pos

def subg_positions(metaG, subgraphs):
    metapos = meta_position(metaG)

    pos_dict = {comm:nx.spring_layout(subgraphs[comm], center=metapos[comm], scale = 0.1) for comm in tqdm(metapos.keys())}

    full_positions = {}

    for p in pos_dict:
        for n in pos_dict[p]:
            full_positions[n] = pos_dict[p][n]

    return full_positions


if __name__ == "__main__":
    G = load_original()
    metaG, subgraphs = make_meta_graph(G)
    full_positions = subg_positions(metaG, subgraphs)

    fig, ax = plt.subplots(figsize=(12,12))
    nx.draw_networkx_nodes(G, pos = full_positions, node_size=2)
    nx.draw_networkx_edges(G, pos=full_positions, node_size=2, width=0.1, alpha = 0.2)

    plt.savefig("community_laid_out.png")
    plt.show()


