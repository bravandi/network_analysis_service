import networkx as nx
import tabulate
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import numpy as np


def convert_bipartite(g):
    b_graph = nx.Graph()
    A = []
    B = []

    for node in g.nodes():
        b_graph.add_node("%i+" % node, bipartite=0)
        b_graph.add_node("%i-" % node, bipartite=1)
        pass

    for edges in g.edges():
        a = "%i+" % edges[0]
        b = "%i-" % edges[1]

        b_graph.add_edge(a, b)

    # not returning correct sets
    # A, B = bipartite.sets(bp)
    A = set(n for n, d in b_graph.nodes(data=True) if d['bipartite'] == 0)
    B = set(b_graph) - A

    return A, B, b_graph


def find_control_nodes(input_graph, remove_loops=False):
    def normalize_node_name(name):
        return int(name.replace('-', '').replace('+', ''))

    print(input_graph.edges())

    if remove_loops is True:
        duplicates = set([e for e in input_graph.edges() if (e[1], e[0]) in input_graph.edges()])
        duplicates = duplicates - set(tuple(sorted(l)) for l in duplicates)
        input_graph.remove_edges_from(duplicates)

    print("Edges: " + str(input_graph.edges()))

    A, B, G = convert_bipartite(input_graph)

    matching_algorithm_result = nx.bipartite.hopcroft_karp_matching(G)

    digraph_matching_edges = [(normalize_node_name(k), normalize_node_name(v)) for k, v in
                              matching_algorithm_result.items() if
                              k.endswith("+")]

    A_vertices_in_matched_edge = set(np.intersect1d(list(matching_algorithm_result.keys()), list(A)).tolist())
    B_vertices_in_matched_edge = set([matching_algorithm_result[m] for m in A_vertices_in_matched_edge])

    # Im not using node names as string because the plot did not work fine
    matched_vertices = set([int(n.replace('-', '')) for n in B_vertices_in_matched_edge])
    unmatched_vertices = set(input_graph.nodes()) - matched_vertices

    control_nodes = unmatched_vertices

    # A_matched_B_matched_intersect = set(np.intersect1d(list(A_vertices_in_matched_edge),
    #                                                    [n.replace('-', '+') for n in B_vertices_in_matched_edge]
    #                                                    ).tolist())
    # A_matched_B_matched_intersect = set()

    # A_vertices_in_matched_edge = A_vertices_in_matched_edge - A_matched_B_matched_intersect
    # B_vertices_in_matched_edge = B_vertices_in_matched_edge - A_matched_B_matched_intersect

    # A_control_nodes = A_vertices_in_matched_edge
    # B_control_candidates = set([m.replace('-', '+') for m in B - B_vertices_in_matched_edge])
    # B_control_nodes = B_control_candidates - A_control_nodes

    # control_nodes = {int(i[:1]) for j in (A_control_nodes, B_control_nodes) for i in j}

    # # # # # # #
    plt.figure("Bipartite Equivalent")
    pos = dict()
    pos.update((n, (1, i)) for i, n in enumerate(sorted(A, reverse=True)))  # put nodes from X at x=1
    pos.update((n, (2, i)) for i, n in enumerate(sorted(B, reverse=True)))  # put nodes from Y at x=2
    edge_colors = []
    for e in G.edges():
        if e[0] in matching_algorithm_result:
            if e == (e[0], matching_algorithm_result[e[0]]):
                edge_colors.append('red')
            else:
                edge_colors.append('black')
        elif e[1] in matching_algorithm_result:
            if e == (matching_algorithm_result[e[1]], e[0]):
                edge_colors.append('red')
            else:
                edge_colors.append('black')
        else:
            edge_colors.append('black')

    sizes = []
    for n in G.nodes():
        sizes.append(1000)

    nx.draw(G, pos=pos, with_labels=True, edge_color=edge_colors, node_size=sizes)

    # # # # # # #
    plt1 = plt.figure("directed graph")
    colors = []
    sizes = []

    for n in input_graph.nodes():
        sizes.append(1000)
        # average_degree_connectivity(G, source='in+out', target='in+out', nodes=None, weight=None)
        if n in control_nodes:
            colors.append('white')
        else:
            colors.append('red')

    nx.draw(input_graph, with_labels=True, node_color=colors, node_size=sizes)
    # # # # # # #

    # # # # # # #
    plt.figure("independent paths")
    graph2 = nx.DiGraph()
    graph2.add_nodes_from(input_graph.nodes())
    graph2.add_edges_from(digraph_matching_edges)

    colors = []
    sizes = []

    for n in input_graph.nodes():
        sizes.append(1000)
        # average_degree_connectivity(G, source='in+out', target='in+out', nodes=None, weight=None)
        if n in control_nodes:
            colors.append('white')
        else:
            colors.append('red')

    nx.draw(graph2, with_labels=True, node_color=colors, node_size=sizes)
    # # # # # # #


    plt.show()

    print("A: %s" % list(A))
    print("B: %s" % list(B))
    # print("Matching Algorithm: %s" % str(matching_algorithm_result))
    print("Digraph matching edges: %s" % str(digraph_matching_edges))
    print("A_matched: %s" % A_vertices_in_matched_edge)
    print("B_matched: %s" % B_vertices_in_matched_edge)
    # print("A_matched_B_matched_intersect: %s" % A_matched_B_matched_intersect)
    # print("A_control_nodes: %s" % A_control_nodes)
    # print("B_control_candidates: %s" % B_control_candidates)
    # print("B_control_nodes: %s" % B_control_nodes)
    print("Len: %i Control nodes: %s" % (len(control_nodes), sorted(control_nodes)))

    return control_nodes


if __name__ == "__main__":
    # print(tabulate.tabulate([["sex", "age"], ["Alice", "F", 24], ["Bob", "M", 19]], headers="firstrow"))

    input_graph = nx.DiGraph()

    input_graph = nx.fast_gnp_random_graph(16, 0.2, directed=True)

    # input_graph.add_edges_from([(0, 1), (0, 4), (1, 3), (2, 0), (2, 6), (5, 2), (6, 3), (6, 4)], directed=True)

    find_control_nodes(input_graph, remove_loops=True)

    # input("press enter to exit")
