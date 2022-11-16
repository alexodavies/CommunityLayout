import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx.algorithms.community as community
from collections import Counter
from tqdm import tqdm
try:
    from datashader.bundling import hammer_bundle
except:
    print(f"COMMUNITY LAYOUT: Datashader not found, edge bundling not available")

def load_original():
    """
    Redundant code for local development. Loads the musae Facebook dataset, avalilable at https://snap.stanford.edu/data/facebook_large.zip

    :return:

    networkx.Graph of the musae facebook graph, without node or edge attributes.
    """
    filename = "../../example_data/musae_facebook_edges.csv"

    social_data = pd.read_csv(filename, delimiter=",", skiprows=0, dtype = str)
    G = nx.from_pandas_edgelist(social_data, "id_1", "id_2", create_using=nx.Graph)

    return G

class CommunityLayout:
    def __init__(self, G,
                 layout_algorithm = nx.spring_layout, layout_kwargs = {"k":75, "iterations":1000},
                 community_compression = 0.25, community_algorithm = community.louvain_communities, community_kwargs = {"resolution":2}):
        """
        Community layout class init.
        Takes user-specified layout and community detection algorithms and lays out a meta-graph and sub-graphs for each community.
        Layout is via the call of .fit()

        :param G: networkx.Graph to be laid out
        :param layout_algorithm: algorithm with the same syntax and returns as a networkx layout algorithm
        :param layout_kwargs: kwargs (e.g. iterations) to be passed to layout algorithm
        :param community_compression: Factor by which to scale-down community sub-layouts
        :param community_algorithm: Algorithm used to partition graph into communities. Should have the same syntax and returns as networkx.algorithm.community algorithms
        :param community_kwargs: kwargs to pass to community algorithm (e.g. resolution)
        """

        try:
            self.G = G
        except:
            raise Exception("No graph passed to init")

        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        self.layout_algorithm = layout_algorithm
        self.layout_kwargs = layout_kwargs
        self.community_compression = community_compression

        self.community_algorithm = community_algorithm
        self.community_kwargs = community_kwargs

        _ = self.fit()

    def get_bundle(self):
        """
        Use datashader.hammer_bundle to perform edge bundling on a set of positions and edges.
        Warning: Much more intensive than the actual node-layout stage. I suggest not using this in its current form.

        :return:

        Dataframe following the syntax of datashader.
        """

        # Prepare dataframes for hammer bundling
        nodes_pos = [[n, *self.full_positions[n]] for n in self.G.nodes()]
        ds_nodes  = pd.DataFrame(nodes_pos, columns = ["name", "x", "y"])

        edge_list = [map(int, edge) for edge in self.G.edges()]
        ds_edges  = pd.DataFrame(edge_list, columns=['source', 'target'])

        hb = hammer_bundle(ds_nodes, ds_edges, batch_size = 40000, iterations = 2)

        # Add bundled dataframe as attribute
        self.hb = hb

    def fit(self):
        """
        Calculate positions of nodes in final layout.
        self.full_positions is set as an attribute in self.subg_positions()

        :return:

        Dictionary of form {"node_id":(x,y),...}
        """

        # Find communities and construct and calculate positions for meta-graph based on their inter-connections
        self.make_meta_graph()

        # Calculate sub-layouts for each community and position according to meta-graph layout
        self.subg_positions()

        # Returns as utility for user, self.full_positions is set as an attribute in self.subg_positions()
        return self.full_positions

    def display(self, colors = None, bundle = False, complex_alphas = True, ax = None):
        """
        Function to produce a visualisation of the community layout result.

        :param colors: Dictionary of form {"node_id":color, ...} with a key for each node in the graph. If not present, colors according to community.
        :param bundle: Whether to apply edge bundling. I advise not to use this in its current form as it is very intensive.
        :param complex_alphas: Whether to give intra-community edges a higher capacity than inter-community
        :param ax: matplotlib.Axes object on which to draw the visualisation.

        :return:

        ax is ax is not None, else saves .png in working directory and calls matplotlib.pyplot.show()
        """


        # Get node colors as a list from dictionary objects.
        if colors is None:
            colors = []
            for n in self.G.nodes():
                colors.append(self.partition[n])
        else:
            colors_list = [colors[n] for n in self.G.nodes()]
            colors = colors_list

        # Whether to perform edge bundling.
        if bundle:
            self.get_bundle()
            self.hb.plot(x="x", y="y", figsize=(9, 9))


        # Get list of edge transparencies.
        edgelist = list(self.G.edges())
        if complex_alphas:
            alphas = []
            for (n1, n2) in edgelist:
                # Different alpha value for intra and inter community edges.
                if self.partition[n1] == self.partition[n2]:
                    alphas.append(0.5)
                else:
                    alphas.append(0.1)
        else:
            alphas = [0.2 for i in range(len(edgelist))]


        # Draw nodes and edges using above and self.full_positions
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 12))
        nx.draw_networkx_nodes(self.G, pos=self.full_positions, node_size=2, ax = ax, node_color=colors)
        nx.draw_networkx_edges(self.G, pos=self.full_positions, node_size=2, width=alphas, alpha=alphas, ax = ax)

        if ax is None:
            plt.savefig("community_laid_out.png")
            plt.show()
        else:
            return ax

    def communities(self):
        """
        Use self.community_algorithm and self.community_kwargs to partition graph into communities.

        :return:

        Partition of node communities, {"community_id":{"node_id1", "node_id2", "node_id3",...}, ...}
        """
        partition = self.community_algorithm(self.G, **self.community_kwargs)
        return partition

    def make_meta_graph(self):
        """
        Use community algorithm to produce meta-graph of communities and intra-links.
        Meta-graph edge weights are the number of inter-community links in original graph.
        """

        # Get community partition of form {"community_id":{"node_id1", "node_id2", "node_id3",...}, ...}
        community_to_node = {i:p for i,p in enumerate(self.communities())}

        # Invert community to node, ie new partition = {"node_id":"community_id", ...}
        partition = {}
        for comm in community_to_node:
            nodes = community_to_node[comm]
            for n in nodes:
                partition[n] = comm

        self.partition = partition


        # Find unique community ids
        community_unique = set([k for k in community_to_node.keys()])

        # Produce a sub-graph for each community
        subgraphs = []
        for c in community_unique:
            subgraphs.append(nx.subgraph(self.G, community_to_node[c]))

        # Get nested list of edges in original graph
        G_edgelist = [[e1, e2] for (e1, e2) in nx.edges(self.G)]

        # Build nested list of edges, of form [["community_id1", "community_id2"], ["community_id3", "community_id4"], ...]
        community_edgelist = []
        for e in G_edgelist:
            comm1 = partition[e[0]]
            comm2 = partition[e[1]]

            community_edgelist.append((comm1, comm2))

        # Find unique edges that are inter-community
        unique_comm_edges = list(set(community_edgelist))
        out_edges = []
        for e in unique_comm_edges:
            if (e[1], e[0]) not in out_edges and e[0] != e[1]:
                out_edges.append(e)
        unique_comm_edges = out_edges

        # Count the number of times each inter-community edge occurs (and the inverse)
        edge_count = [community_edgelist.count(e) + community_edgelist.count([e[1], e[0]]) for e in unique_comm_edges]

        # Package inter-community edges and their counts as a list of tuples, [("community_id1", "community_id2", count),...]
        full_description = [(*list(unique_comm_edges)[i], edge_count[i]) for i in range(len(edge_count))]

        # Build metagraph as a weighted networkx graph
        metaG = nx.Graph()
        metaG.add_weighted_edges_from(full_description)

        # Set metagraph and community subgraphs as attributes
        self.subgraphs = {i:g for i, g in enumerate(subgraphs)}
        self.metaG = metaG

    def meta_position(self, show = False):
        """
        Calulate layout of meta-graph. Layout algorithm should take edge weight as a parameter.

        :param show: Whether to show the resulting layout. Not called in default usage, but left as a utility for user.
        """

        # Calculate layout using self.layout_algorithm and self.layout_kwargs from __init__
        pos = self.layout_algorithm(self.metaG, weight="weight", **self.layout_kwargs)
        print(f"Metagraph is a {self.metaG}")

        if show:
            fig, ax = plt.subplots(figsize=(12,12))

            weights = np.array(list(nx.get_edge_attributes(self.metaG,'weight').values()))
            weights = weights / np.max(weights)
            nx.draw_networkx_edges(self.metaG, pos, node_size=20, ax = ax, width=5*weights + 0.1)
            nx.draw_networkx_nodes(self.metaG, pos, node_size=20, ax=ax)


            plt.savefig("metagraph.png")
            plt.close()

        #Set metagraph layout as class attribute
        self.metapos = pos

    def subg_positions(self):
        """
        Calculate layouts for each community subgraph. Position center according to meta-graph layout.
        """

        self.meta_position(self.metaG)

        pos_dict = {comm:self.layout_algorithm(self.subgraphs[comm],
                                               center=self.metapos[comm],
                                               scale = self.community_compression)
                    for comm in tqdm(self.metapos.keys())}

        full_positions = {}

        for p in pos_dict:
            for n in pos_dict[p]:
                full_positions[n] = pos_dict[p][n]

        self.full_positions = full_positions



