import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx.algorithms.community as community
from collections import Counter
from tqdm import tqdm
# from datashader.bundling import hammer_bundle


def load_original():
    filename = "../example_data/musae_facebook_edges.csv"

    social_data = pd.read_csv(filename, delimiter=",", skiprows=0, dtype = str)
    G = nx.from_pandas_edgelist(social_data, "id_1", "id_2", create_using=nx.Graph)

    return G

class CommunityLayout:
    def __init__(self, G,
                 layout_algorithm = nx.spring_layout, layout_kwargs = {"weight":"weight", "k":75, "iterations":1000},
                 community_compression = 0.25):
        try:
            self.G = G
        except:
            raise Exception("No graph passed to init")

        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        self.layout_algorithm = layout_algorithm
        self.layout_kwargs = layout_kwargs
        self.community_compression = community_compression

        _ = self.fit()

    def get_bundle(self):

        nodes_pos = [[n, *self.full_positions[n]] for n in self.G.nodes()]
        ds_nodes  = pd.DataFrame(nodes_pos, columns = ["name", "x", "y"])

        print(ds_nodes)

        edge_list = [map(int, edge) for edge in self.G.edges()]
        ds_edges  = pd.DataFrame(edge_list, columns=['source', 'target'])

        print(ds_edges)

        hb = hammer_bundle(ds_nodes, ds_edges, batch_size = 40000, iterations = 2)

        self.hb = hb

    def fit(self):
        self.make_meta_graph()
        self.subg_positions()

        return self.full_positions

    def display(self, colors = None, bundle = False, complex_alphas = True):

        if colors is None:
            colors = []
            for n in self.G.nodes():
                colors.append(self.partition[n])

        if bundle:
            self.get_bundle()
            self.hb.plot(x="x", y="y", figsize=(9, 9))

        edgelist = list(self.G.edges())
        if complex_alphas:

            alphas = []

            for (n1, n2) in edgelist:
                if self.partition[n1] == self.partition[n2]:
                    alphas.append(0.5)
                else:
                    alphas.append(0.1)
        else:
            alphas = [0.2 for i in range(len(edgelist))]


        fig, ax = plt.subplots(figsize=(12, 12))
        nx.draw_networkx_nodes(self.G, pos=self.full_positions, node_size=2, ax = ax, node_color=colors)
        nx.draw_networkx_edges(self.G, pos=self.full_positions, node_size=2, width=alphas, alpha=alphas, ax = ax)

        plt.savefig("community_laid_out.png")
        plt.show()

    def communities(self, G):
        partition = community.louvain_communities(G, resolution=2)
        return partition

    def make_meta_graph(self):
        community_to_node = {i:p for i,p in enumerate(self.communities(self.G))}
        partition = {}
        for comm in community_to_node:
            nodes = community_to_node[comm]
            for n in nodes:
                partition[n] = comm

        self.partition = partition


        community_unique = set([k for k in community_to_node.keys()])
        subgraphs = []

        for c in community_unique:
            subgraphs.append(nx.subgraph(self.G, community_to_node[c]))

        G_edgelist = [[e1, e2] for (e1, e2) in nx.edges(self.G)]

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
        metaG = nx.Graph()


        metaG.add_weighted_edges_from(full_description)

        self.subgraphs = {i:g for i, g in enumerate(subgraphs)}

        self.metaG = metaG

    def meta_position(self, show = True):

        # pos = nx.spring_layout(self.metaG, weight="weight", k=100, iterations=1000)
        pos = self.layout_algorithm(self.metaG, **self.layout_kwargs)
        print(f"Metagraph is a {self.metaG}")

        if show:
            fig, ax = plt.subplots(figsize=(12,12))

            weights = np.array(list(nx.get_edge_attributes(self.metaG,'weight').values()))
            weights = weights / np.max(weights)
            nx.draw_networkx_edges(self.metaG, pos, node_size=20, ax = ax, width=5*weights + 0.1)
            nx.draw_networkx_nodes(self.metaG, pos, node_size=20, ax=ax)


            plt.savefig("metagraph.png")
            plt.close()
        self.metapos = pos

    def subg_positions(self):
        self.meta_position(self.metaG)


        pos_dict = {comm:self.layout_algorithm(self.subgraphs[comm], center=self.metapos[comm], scale = self.community_compression) for comm in tqdm(self.metapos.keys())}

        full_positions = {}

        for p in pos_dict:
            for n in pos_dict[p]:
                full_positions[n] = pos_dict[p][n]

        self.full_positions = full_positions



