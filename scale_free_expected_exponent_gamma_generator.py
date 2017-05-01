import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def scale_free_from_bimodal_paper(
        n,
        alpha_in,
        alpha_out,
        k):
    # probability proportional to w_i and w_j , respectively.
    def weighted_choice(choices):
        total = sum([w for c, w in choices])
        r = random.uniform(0, total)

        upto = 0
        for c, w in choices:
            if upto + w >= r:
                return c
            upto += w
        assert False, "Shouldn't get here"

    if alpha_out == 0:
        gamma_out = 0
    else:
        gamma_out = 1 + (1 / alpha_out)
    if alpha_in == 0:
        gamma_in = 0
    else:
        gamma_in = 1 + (1 / alpha_in)

    print ("gamma_out: {} gamma_in: {}".format(gamma_out, gamma_in))

    G = nx.DiGraph()
    nodes = range(1, int(n) + 1)
    G.add_nodes_from(nodes)

    w_in = []
    w_out = []

    # assign a weight or expected degree w_i^(out,in) = i^(-1*(alpha_out,alpha_in))
    # to each node in the out and the in set,
    for i in range(1, int(n) + 1):
        w_in.append(pow(float(i), (-1 * alpha_in)))
        w_out.append(pow(float(i), (-1 * alpha_out)))

    edges_num = 0
    n_links = n * k / 2

    ch_in = zip(nodes, w_in)
    ch_out = zip(nodes, w_out)

    selected_in = {}
    selected_out = {}

    # add links until we get the expected n_links
    while edges_num != n_links:
        out_node = weighted_choice(ch_in)
        in_node = weighted_choice(ch_out)

        if in_node not in selected_in:
            selected_in[in_node] = 1
        else:
            selected_in[in_node] += 1

        if out_node not in selected_out:
            selected_out[out_node] = 1
        else:
            selected_out[out_node] += 1

        G.add_edge(out_node, in_node)

        G.remove_edges_from(G.selfloop_edges())

        edges_num = len(G.edges())

    num_nodes = len(G.nodes())
    num_links = len(G.edges())
    print ("number of nodes: %i  number of link: %i <k>: %f" %
           (num_nodes, num_links, round(num_links / float(num_nodes), 2)))

    return G


if __name__ == "__main__":
    # test_scale_free()
    k_avg = 5
    n = 700
    alpha_out = 0.6
    alpha_in = 0.5

    G = scale_free_from_bimodal_paper(n=n, alpha_out=alpha_out, alpha_in=alpha_in, k=k_avg)

    if G.number_of_selfloops() > 0:
        print ("SELF LOOP")

    print(
        "      out var: {} in var: {}".format(str(np.var(G.out_degree().values())),
                                              np.var(G.in_degree().values())))

    plt.hist(G.in_degree().values(), 50, normed=True, color='green')
    plt.hist(G.out_degree().values(), 50, normed=True, color='orange')
    plt.title("out-deg (orange) in-deg(green) distributions")
    plt.show()

    # ;;;;;;;save the network if needed ;;;;;;;;;;;;
    # nx.write_gml(G, "d:\\temp\\scale_free_bimodal\\"
    #                 "scale_free_k_{}_n_{}_gamma_out_{}_gamma_in_{}_.gml".format(
    #     k_avg, n,
    #     round(1 + (1 / alpha_out), 3),  # gamma_out
    #     round(1 + (1 / alpha_in), 3)  # gamma_in
    # ))
    print ("-------------Scale free - DONE-----------------------")
