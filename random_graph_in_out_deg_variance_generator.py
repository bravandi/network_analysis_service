import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def random_network_given_degree_sequence(n):
    show_plots = True

    deg_seq_length = n

    out_deg_seq = np.random.poisson(6, deg_seq_length).tolist()
    in_deg_seq = np.random.poisson(5, deg_seq_length).tolist()

    # in order to increase the number of links, therefore, <k>
    for i in range(0, len(out_deg_seq)):
        out_deg_seq[i] += 5
    for i in range(0, len(in_deg_seq)):
        in_deg_seq[i] += 8

    if show_plots:
        plt.hist(in_deg_seq, 50, normed=True, alpha=0.75, color='green')
        plt.hist(out_deg_seq, 52, normed=True, color='orange')
        plt.title("Sequences generated for out-deg (orange) in-deg (green)")
        plt.show()

    # the sum of in-deg and out-deg must be equal
    while True:
        in_deg_seq_sum = sum(in_deg_seq)
        out_deg_seq_sum = sum(out_deg_seq)
        if in_deg_seq_sum == out_deg_seq_sum:
            break

        # to have different variance in out and in degree distributions
        add_to = 5
        if abs(in_deg_seq_sum - out_deg_seq_sum) < add_to:
            add_to = 1

        if in_deg_seq_sum > out_deg_seq_sum:
            i = random.choice(range(0, len(out_deg_seq)))
            out_deg_seq[i] += add_to
        else:
            i = random.choice(range(0, len(in_deg_seq)))
            in_deg_seq[i] += add_to

    G = nx.directed_configuration_model(in_deg_seq, out_deg_seq)
    print ("number of self loops: " + str(G.number_of_selfloops()))
    G.remove_edges_from(G.selfloop_edges())

    # from Multi link Directed Graph convert to Directed Graph.
    # However, initially it does not contain multi links between two nodes
    G = G.to_directed()

    if show_plots:
        plt.hist(G.in_degree().values(), 50, normed=True, color='green')
        plt.hist(G.out_degree().values(), 50, normed=True, color='orange')
        plt.title("out-deg (orange) in-deg(green) distributions of the graph")
        plt.show()

    num_nodes = len(G.nodes())
    num_links = len(G.edges())
    k_avg = num_links / float(num_nodes)

    print (
        "number of nodes: %i  link#: %i <k>: %f" % (num_nodes, num_links, round(k_avg, 2)))

    print(
        "p: {} Degree Distributions: [OUT-deg variance: {}] [in-deg variance: {} ] out and in degree mean: {}".format(
            k_avg / (num_nodes - 1),
            np.var(G.out_degree().values()),
            np.mean(G.out_degree().values()),
            np.var(G.in_degree().values()),
            np.mean(G.in_degree().values())
        ))

    return G


if __name__ == "__main__":
    random_network_given_degree_sequence(n=1000)

    print ("------------------------------------------")
