import random
from Tkinter import Tk
from complex_networks.network import Network
from complex_networks.experiment import Experiment
import snap
import complex_networks.constants as constants
import complex_networks.tools as tools
from datetime import datetime
import networkx as nx
import complex_networks.networkx_gml as networkx_gml
import complex_networks.graph_generator as graph_generator
import matplotlib.pyplot as plt
import bisect
import numpy as np
import math
import os
from os import walk
from shutil import copyfile


def network_degree_distribution_from_path(path):
    net = Network.networkx_create_from_gml(
        path=path
        # path="D:\\SoftwareProject\\complex_networks_tools\\data\\Neural Network\\celegansneural.gml"
    )

    in_degree, out_degree, total_degree = Network.nx_degree_distribution(net)

    num_nodes = len(net.nodes())
    num_links = len(net.edges())
    print (path)
    print ("number of nodes: %i  link#: %i <k>: %f" % (num_nodes, num_links, round(num_links / num_nodes, 2)))
    print ("in_degree: %s \nout_degree %s \n total_degree %s" % (in_degree, out_degree, total_degree))


def wtf():
    def p_out(x):

        if x == 0:
            return 3.0 / 5.0
        if x == 2:
            return 2.0 / 5.0

        # if x == 1 or x == 2:
        #     return 2.0 / 5.0

        return 0

    def p_in(x):

        if x == 0:
            return 2.0 / 5.0
        if x == 1:
            return 2.0 / 5
        if x == 2:
            return 1.0 / 5.0

        return 0

    def in_deg(x):
        if x == 3.0:
            return 1
        if x == 4.0:
            return 2
        if x == 5.0:
            return 1
        return 0.0

    def out_deg(x):
        if x == 1 or x == 2:
            return 2.0

        return 0.0

    h_in = 0
    k = (4.0 / 5.0)

    for i in range(0, 3):
        for j in range(0, 3):
            # h_in += abs(in_deg(i) - in_deg(j)) * ((p_in(in_deg(i)) * p_in(in_deg(j))) / k)
            # h_in += abs(i - j) * ((p_in(in_deg(i)) * p_in(in_deg(j))) / k)
            h_in += abs(i - j) * ((p_in(i) * p_in(j)) / k)

    print ("h_in: " + str(h_in))

    h_out = 0
    h_out += abs(i - j) * ((p_out(out_deg(i)) * p_out(out_deg(j))) / k)

    # for i in range(0, 3):
    #     for j in range(0, 3):
    #         # h_out += abs(i - j) * ((p_out(out_deg(i)) * p_out(out_deg(j))) / k)
    #         # h_out += abs(out_deg(i) - out_deg(j)) * ((p_out(out_deg(i)) * p_out(out_deg(j))) / k)
    #         if (i == 1 and j == 3) or (i == 1 and j == 4) or (i == 2 and j == 4) or (i == 2 and j == 5):
    #             # h_out += (abs(i - j) * p_out(i) * p_out(j)) / k
    #             h_out += abs(i - j) * ((p_out(out_deg(i)) * p_out(out_deg(j))) / k)

    print ("h_out:" + str(h_out))


def create_experiment_network(initialize):
    ex = Experiment(debug=True, draw_graphs=True)

    network = ex.snap_load_network(
        # graph_path='data/High-energy physics theory citation network/Cit-HepTh.txt',
        # network_id=1212,
        # name='physics citation networks',

        # graph_path='data/Amazon product co-purchasing network, March 02 2003/Amazon0302.txt',
        # network_id=1,
        # name='Amazon product co-purchasing network, March 02 2003',

        graph_path='data/small_sample.txt',
        network_id=0,
        name='small_sample',

        initialize_graph=initialize,
        model=constants.NetworkModel.real_network(),

    )

    return ex, network


def find_control_nodes():
    # start_time = datetime.now()
    # print ("\nstart_time: %s\n" % str(datetime.now()))

    ex, network = create_experiment_network(initialize=True)

    network.control.snap_find_mds_minimum_driver_node_set()

    net2 = ex.snap_to_networkx_cnetwork(snap_g=network.graph, name=network.name, network_id=network.network_id,
                                        model=network.model)

    nx.write_gml(net2.graph, path="d:\\temp\\test.gml")


    # print ("\n\nend time: %s\nduration: %s" %
    #        (
    #            str(datetime.now()),
    #            str(tools.get_time_difference(start_time, datetime.now()))
    #        ))


def bfs():
    source_node = 1

    G1 = snap.PUNGraph.New()
    bfs_tree = snap.PUNGraph.New()

    bfs_tree.AddNode(source_node)

    for i in range(1, 8 + 1):
        G1.AddNode(i)

    G1.AddEdge(1, 6)
    G1.AddEdge(2, 5)
    G1.AddEdge(2, 6)
    G1.AddEdge(3, 6)
    G1.AddEdge(3, 7)
    G1.AddEdge(3, 8)
    G1.AddEdge(4, 7)

    vec = snap.TIntV()

    vec.Add(source_node)

    while not vec.Empty():

        u = vec[0]
        vec.Del(0)

        for v in G1.GetNI(u).GetOutEdges():

            if bfs_tree.IsNode(v) is False:
                bfs_tree.AddNode(v)
                bfs_tree.AddEdge(u, v)

                vec.Add(v)
    # snap.GetShortPath_PUNGraph(bfs_tree, 1, 10)

    print ([(e.GetSrcNId(), e.GetDstNId()) for e in bfs_tree.Edges()])


def find_redundant_nodes(initialize):
    ex, network = create_experiment_network(initialize)

    network.control.snap_find_redundant_intermittent_critical_nodes()


def developing_netlogo_communication():
    net = Network.networkx_create_from_gml(
        path="d:\\temp\\save-gml.gml"
        # path="D:\\SoftwareProject\\complex_networks_tools\\data\\Neural Network\\celegansneural.gml"
    )

    ex = Experiment(debug=True)

    cnetwork = ex.networkx_to_snap_cnetwork(
        networkx=net,
        name="",
        model=constants.NetworkModel.netlogo(),
        network_id=100
    )

    unmatched_nodes_inset, mds = cnetwork.control.snap_find_mds_minimum_driver_node_set()

    redundant_nodes, intermittent_nodes, critical_nodes = \
        cnetwork.control.snap_find_redundant_intermittent_critical_nodes()

    redundant_nodes_who = ''
    intermittent_nodes_who = ''
    critical_nodes_who = ''
    mds_who = ''

    for node in redundant_nodes:
        redundant_nodes_who += dict(net.nodes(True))[node]['WHO'][:-2] + ' '

    for node in intermittent_nodes:
        intermittent_nodes_who += dict(net.nodes(True))[node]['WHO'][:-2] + ' '

    for node in critical_nodes:
        critical_nodes_who += dict(net.nodes(True))[node]['WHO'][:-2] + ' '

    for node in mds:
        mds_who += dict(net.nodes(True))[node]['WHO'][:-2] + ' '

    result = "r:[%s]:r i:[%s]:i c:[%s]:c mds:[%s]:mds" % \
             (
                 redundant_nodes_who,
                 intermittent_nodes_who,
                 critical_nodes_who,
                 mds_who
             )

    r = Tk()
    r.withdraw()
    r.clipboard_clear()
    r.clipboard_append('color-code-control-nodes "%s"' % result)
    r.destroy()

    print (result)


# deg_tri = [
#     [1, 0], [1, 0], [1, 0], [2, 0], [1, 0], [2, 1], [0, 1], [0, 1],
#     [1, 0], [1, 0], [1, 0], [2, 0], [1, 0], [2, 1], [0, 1], [0, 1],
#     [1, 0], [1, 0], [1, 0], [2, 0], [1, 0], [2, 1], [0, 1], [0, 1],
#     [1, 0], [1, 0], [1, 0], [2, 0], [1, 0], [2, 1], [0, 1], [0, 1],
#            ]
# G = nx.random_clustered_graph(deg_tri)
# G = nx.Graph(G)


def networkx_gml_test():
    G = nx.scale_free_graph(200, delta_in=1, delta_out=0, alpha=0.05, beta=0.45, gamma=0.5)
    # G = nx.grid_2d_graph(10, 10)
    # G = nx.cycle_graph(50)
    # networkx_gml.read_gml(path, label='id')
    # G = nx.fast_gnp_random_graph(100, 0.3)

    # G = nx.gnp_random_graph(100, 0.02, directed=True)
    # G = nx.gnm_random_graph(100, 200, directed=True)
    # G = nx.binomial_graph(100, 0.01, directed=True)
    # in_degree, out_degree, total_degree = Network.nx_degree_distribution(G)

    # G = graph_generator.gnm_random_graph(200, 400, directed=True)
    # G = nx.erdos_renyi_graph(200, 0.005, directed=True)
    Network.nx_reverse_links(G)

    # G = nx.gn_graph(50)

    # G.remove_edges_from(G.selfloop_edges())

    num_nodes = len(G.nodes())
    num_links = len(G.edges())
    print ("number of nodes: %i  link#: %i <k>: %f" % (num_nodes, num_links, round(num_links / float(num_nodes), 2)))

    networkx_gml.write_gml(G, "d:\\temp\\test.gml")

    net = Network.networkx_create_from_gml(
        path="d:\\temp\\test.gml"
        # path="D:\\SoftwareProject\\complex_networks_tools\\data\\Neural Network\\celegansneural.gml"
    )

    ex = Experiment(debug=True)

    cnetwork = ex.networkx_to_snap_cnetwork(
        networkx=net,
        name="",
        model=constants.NetworkModel.netlogo(),
        network_id=100
    )

    in_degree, out_degree, total_degree = cnetwork.snap_degree_distribution()

    print (in_degree)
    print (out_degree)

    unmatched_nodes_inset, mds = cnetwork.control.snap_find_mds_minimum_driver_node_set()

    redundant_nodes, intermittent_nodes, critical_nodes = \
        cnetwork.control.snap_find_redundant_intermittent_critical_nodes()

    print("")


def scale_free():
    def node_added(G):
        num_nodes = len(G.nodes())
        num_links = len(G.edges())

        # print (round(num_links / float(num_nodes), 2))
        # print (G)

    G2 = graph_generator.scale_free_graph(
        2000, delta_in=1, delta_out=0, alpha=0.05, beta=0.9, gamma=0.05,
        node_added=node_added
    )

    nx.draw_networkx_nodes(G2, pos=nx.spring_layout(G2))

    num_nodes = len(G2.nodes())
    num_links = len(G2.edges())
    print ("number of nodes: %i  link#: %i <k>: %f" % (num_nodes, num_links, round(num_links / float(num_nodes), 2)))
    # nx.draw(G2)

    pos = nx.spring_layout(G2)
    f1 = plt.figure(3, figsize=(12, 12))
    nx.draw_networkx_nodes(G2, pos, cmap=plt.get_cmap('jet'))
    # nx.draw_networkx_edges(G2, pos, edgelist=red_edges, edge_color='r', arrows=True)
    # nx.draw_networkx_edges(G2, pos, edgelist=black_edges, arrows=False)
    nx.draw_networkx_edges(G2, pos, edgelist=G2.edges(), arrows=True)

    print ([nid for nid, val in G2.degree().items() if val == 0])

    # plt.show()


def scale_free_hidden_parameter_from_network_science_book(
        n,
        alpha_in,
        alpha_out,
        k_mean):
    e = 2.71828182  # 84590452353602874713527

    def weighted_choice(choices, total):
        # total = sum([w for c, w in choices])
        r = random.uniform(0, total)

        upto = 0
        for c, w in choices:
            if upto + w >= r:
                return c
            upto += w
        assert False, "Shouldn't get here"

    c = 1.0
    n_i = [c / pow(i, alpha_out) for i in range(1, n + 1)]
    # n_i.insert(0, 0)

    n_i_avg = sum(n_i) / float(n)

    # dist = {}
    # for k in range(1, 100):
    #     dist[k] = p_k(k, n_i)

    lambda_out = 1 + (1 / alpha_out)
    lambda_in = 1 + (1 / alpha_in)

    print ("lambda_out: {} lambda_in: {}".format(
        lambda_out,
        lambda_in
    ))

    G = nx.DiGraph()
    nodes = range(1, n + 1)
    G.add_nodes_from(nodes)

    ch_in = zip(nodes, n_i)

    edges_num = 0.0
    tst = []
    total = sum(n_i)

    while edges_num / float(n) < k_mean:
        i = weighted_choice(ch_in, total=total)
        j = weighted_choice(ch_in, total=total)
        G.add_edge(i, j)

        edges_num = len(G.edges())

    return G


def experiment_scale_free_hidden_parameter_from_network_science_book(show_plots=True):
    ex = Experiment(debug=False, draw_graphs=False)

    for i in range(0, 11):
        start_time_2 = datetime.now()

        G = scale_free_hidden_parameter_from_network_science_book(
            n=1000,
            alpha_out=0.5,
            alpha_in=-1,
            k_mean=i + 6
        )

        if show_plots:
            count, bins, ignored = plt.hist(G.in_degree().values(), 50, normed=True)
            plt.title("in-deg from graph")
            # plt.show()
            count, bins, ignored = plt.hist(G.out_degree().values(), 50, normed=True, color='orange')
            plt.title("out-deg from graph")
            plt.show()

        print ("number of self loops: " + str(G.number_of_selfloops()))
        G.remove_edges_from(G.selfloop_edges())

        nx.write_gml(G, "d:\\temp\\test2.gml")

        num_nodes = len(G.nodes())
        num_links = len(G.edges())
        k_avg = num_links / float(num_nodes)
        print (
            "number of nodes: %i  link#: %i <k>: %f" % (num_nodes, num_links, round(k_avg, 2)))

        print(
            "p: {} [OUT variance: {} mean: {}] [in variance: {} mean: {}]".format(
                k_avg / (num_nodes - 1),
                np.var(G.out_degree().values()),
                np.mean(G.out_degree().values()),
                np.var(G.in_degree().values()),
                np.mean(G.in_degree().values())
            ))

        network_cn = ex.networkx_to_snap_cnetwork(networkx=G, name="", network_id=100 + i,
                                                  model=constants.NetworkModel.real_network())

        unmatched_nodes_inset, mds = network_cn.control.snap_find_mds_minimum_driver_node_set()
        redundant_nodes, intermittent_nodes, critical_nodes = \
            network_cn.control.snap_find_redundant_intermittent_critical_nodes()

        print ("@@redundant nodes ratio: {} <k>: {} time took to process: {} seconds".format(
            float(len(redundant_nodes)) / len(G.nodes()),
            round(k_avg, 2),
            tools.get_time_difference_seconds(start_time_2)
        ))
        print ("------------------------------------------")


def load_network_from_text(path):
    ex = Experiment(debug=False, draw_graphs=False)

    network_cn = ex.snap_load_network(

        graph_path=path,
        network_id=0,
        name='small_sample',

        initialize_graph=True,
        model=constants.NetworkModel.real_network(),

    )

    # print("out var: {} in var: {}".format(str(np.var(G.out_degree().values())), np.var(G.in_degree().values())))

    unmatched_nodes_inset, mds = network_cn.control.snap_find_mds_minimum_driver_node_set()
    redundant_nodes, intermittent_nodes, critical_nodes = \
        network_cn.control.snap_find_redundant_intermittent_critical_nodes()

    netx = ex.snap_to_networkx_cnetwork(snap_g=network_cn.graph, name=network_cn.name, network_id=network_cn.network_id)
    G = netx.graph

    num_nodes = float(len(G.nodes()))
    num_links = float(len(G.edges()))
    print ("number of nodes: %i  link#: %i <k>: %f" % (num_nodes, num_links, round(num_links / float(num_nodes), 2)))

    output_file_name = "k_{0}_n_{1}_e_{2}".format(
        round(num_links / num_nodes, 3), num_nodes, num_links)

    tools.clipboard_copy("fromR\\\\" + output_file_name)

    nx.write_gml(G, "d:\\temp\\fromR\\" + output_file_name + ".gml")

    print ("d:\\temp\\fromR\\" + output_file_name + ".gml")

    print (float(len(redundant_nodes)) / network_cn.graph.GetNodes())


def scale_free_from_bimodal_paper(
        n,
        alpha_in,
        alpha_out,
        k):
    def weighted_choice_cumilative_sums(choices, cumilative_sums):
        # cumilative_sums = np.cumsum([w for c, w in choices]).tolist()
        rnd = random.uniform(0, cumilative_sums[len(cumilative_sums) - 1])

        i = bisect.bisect(cumilative_sums, rnd)
        return choices[i][0], i

    def weighted_choice(choices):
        total = sum([w for c, w in choices])
        r = random.uniform(0, total)

        upto = 0
        for c, w in choices:
            if upto + w >= r:
                return c
            upto += w
        assert False, "Shouldn't get here"

    def np_weighted_choice(choices, mean, std):
        r = abs(np.random.normal(mean, std, 1)[0])

        upto = 0
        while True:
            item = random.choice(choices)
            if item[1] >= r:
                return item[0]

        assert False, "Shouldn't get here"

    if alpha_out == 0:
        lambda_out = 0
    else:
        lambda_out = 1 + (1 / alpha_out)
    if alpha_in == 0:
        lambda_in = 0
    else:
        lambda_in = 1 + (1 / alpha_in)

    print ("lambda_out: {} lambda_in: {}".format(
        lambda_out,
        lambda_in
    ))

    G = nx.DiGraph()
    nodes = range(1, int(n) + 1)
    G.add_nodes_from(nodes)

    w_in = []
    w_out = []
    w_in_mean = None
    w_in_std = None
    w_out_mean = None
    w_out_std = None

    for i in range(1, int(n) + 1):
        w_in.append(pow(float(i), (-1 * alpha_in)))
        w_out.append(pow(float(i), (-1 * alpha_out)))

    w_in_mean = np.mean(w_in)
    w_in_std = np.std(w_in)
    w_out_mean = np.mean(w_out)
    w_out_std = np.std(w_out)

    w_in_cumilative_sums = np.cumsum(w_in).tolist()
    w_out_cumilative_sums = np.cumsum(w_out).tolist()

    # import matplotlib.pyplot as plt
    # count, bins, ignored = plt.hist( abs(np.random.normal(w_in_mean, w_in_std, 1000)), 30, normed=True)
    # plt.plot(bins, 1 / (w_in_std * np.sqrt(2 * np.pi)) *
    #          np.exp(- (bins - w_in_mean) ** 2 / (2 * w_in_std ** 2)),
    #          linewidth=2, color='r')
    # plt.show()

    edges_num = 0

    ch_in = zip(nodes, w_in)
    ch_out = zip(nodes, w_out)

    selected_in = {}
    selected_out = {}

    while edges_num / n != k:
        # ch_in_node, ch_in_index = weighted_choice_cumilative_sums(ch_in, w_in_cumilative_sums)
        # ch_out_node, ch_out_index = weighted_choice_cumilative_sums(ch_out, w_out_cumilative_sums)

        # in_node, ch_in_index = weighted_choice_cumilative_sums(ch_in, w_in_cumilative_sums)
        # out_node, ch_out_index = weighted_choice_cumilative_sums(ch_out, w_out_cumilative_sums)

        # if w_in[ch_in_index] > w_out[ch_out_index]:
        #     out_node = ch_in_node
        #     in_node = ch_out_node
        #
        # else:
        #     in_node = ch_in_node
        #     out_node = ch_out_node

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

        # in_node = np_weighted_choice(ch_in, w_in_mean, w_in_std)
        # out_node = np_weighted_choice(ch_out, w_out_mean, w_out_std)

        # if G.has_edge(out_node, in_node) is False:
        G.add_edge(out_node, in_node)

        G.remove_edges_from(G.selfloop_edges())

        edges_num = len(G.edges())

    num_nodes = len(G.nodes())
    num_links = len(G.edges())
    print ("number of nodes: %i  link#: %i <k>: %f" % (num_nodes, num_links, round(num_links / float(num_nodes), 2)))

    output_file_name = "alphaOut_{0}_alphaIn_{1}_k_{2}_n_{3}_e_{4}".format(
        round(lambda_out, 3), round(lambda_in, 3), k, num_nodes, num_links)

    tools.clipboard_copy("scale_free_bimodal\\\\" + output_file_name)

    nx.write_gml(G, "d:\\temp\\scale_free_bimodal\\" + output_file_name + ".gml")

    print ("       d:\\temp\\scale_free_bimodal\\" + output_file_name + ".gml")

    return G


def experiment_scale_free_from_bimodal_paper(show_plots=True):
    load_network_from_text(path="d:\\temp\\g.txt")

    ex = Experiment(debug=False, draw_graphs=False)

    p = 0

    for k_avg in range(6, 10):
        start_time_2 = datetime.now()

        G = scale_free_from_bimodal_paper(
            n=800,
            # alpha_out=0.5,
            # alpha_in=0.6,
            alpha_out=0.6,
            alpha_in=0.5,
            k=k_avg
        )

        if show_plots:
            count, bins, ignored = plt.hist(G.in_degree().values(), 50, normed=True)
            plt.title("in-deg from graph")
            # plt.show()
            count, bins, ignored = plt.hist(G.out_degree().values(), 50, normed=True, color='orange')
            plt.title("out-deg from graph")
            plt.show()

        if G.number_of_selfloops() > 0:
            print ("SELF LOOP")

        print(
            "      out var: {} in var: {}".format(str(np.var(G.out_degree().values())), np.var(G.in_degree().values())))

        network_cn = ex.networkx_to_snap_cnetwork(networkx=G, name="", network_id=100 + k_avg,
                                                  model=constants.NetworkModel.real_network())

        unmatched_nodes_inset, mds = network_cn.control.snap_find_mds_minimum_driver_node_set()
        redundant_nodes, intermittent_nodes, critical_nodes = \
            network_cn.control.snap_find_redundant_intermittent_critical_nodes()

        print ("@@redundant nodes ratio: {} <k>: {} time took to process: {} seconds".format(
            float(len(redundant_nodes)) / len(G.nodes()),
            round(k_avg, 2),
            tools.get_time_difference_seconds(start_time_2)
        ))
        print ("------------------------------------------")


def random_network_given_degree_sequence():
    ex = Experiment(debug=False, draw_graphs=False)
    show_plots = True

    deg_seq_length = 500
    for i in range(0, 11):
        start_time_2 = datetime.now()

        deg_seq_length += 200
        out_deg_seq = np.random.poisson(2 + i, deg_seq_length).tolist()
        in_deg_seq = np.random.poisson(4 + i, deg_seq_length).tolist()

        if show_plots:
            count, bins, ignored = plt.hist(in_deg_seq, 50, normed=True, alpha=0.75)
            plt.title("in-deg seq")
            # plt.show()
            count, bins, ignored = plt.hist(out_deg_seq, 52, normed=True, color='orange')
            plt.title("out-deg seq")
            plt.show()

        while True:
            in_deg_seq_sum = sum(in_deg_seq)
            out_deg_seq_sum = sum(out_deg_seq)
            if in_deg_seq_sum == out_deg_seq_sum:
                break

            add_to = 10
            if abs(in_deg_seq_sum - out_deg_seq_sum) < add_to:
                add_to = 1

            if in_deg_seq_sum > out_deg_seq_sum:
                i = random.choice(range(0, len(out_deg_seq)))
                out_deg_seq[i] += add_to
            else:
                i = random.choice(range(0, len(in_deg_seq)))
                in_deg_seq[i] += add_to

        # G = nx.directed_havel_hakimi_graph(in_deg_seq, out_deg_seq)
        G = nx.directed_configuration_model(in_deg_seq, out_deg_seq)
        G = G.to_directed()

        if show_plots:
            count, bins, ignored = plt.hist(G.in_degree().values(), 50, normed=True)
            plt.title("in-deg from graph")
            # plt.show()
            count, bins, ignored = plt.hist(G.out_degree().values(), 50, normed=True, color='orange')
            plt.title("out-deg from graph")
            plt.show()

        print ("number of self loops: " + str(G.number_of_selfloops()))
        G.remove_edges_from(G.selfloop_edges())

        nx.write_gml(G, "d:\\temp\\test2.gml")

        num_nodes = len(G.nodes())
        num_links = len(G.edges())
        k_avg = num_links / float(num_nodes)
        print (
            "number of nodes: %i  link#: %i <k>: %f" % (num_nodes, num_links, round(k_avg, 2)))

        print(
            "p: {} [OUT variance: {} mean: {}] [in variance: {} mean: {}]".format(
                k_avg / (num_nodes - 1),
                np.var(G.out_degree().values()),
                np.mean(G.out_degree().values()),
                np.var(G.in_degree().values()),
                np.mean(G.in_degree().values())
            ))

        network_cn = ex.networkx_to_snap_cnetwork(networkx=G, name="", network_id=100 + i,
                                                  model=constants.NetworkModel.real_network())

        unmatched_nodes_inset, mds = network_cn.control.snap_find_mds_minimum_driver_node_set()
        redundant_nodes, intermittent_nodes, critical_nodes = \
            network_cn.control.snap_find_redundant_intermittent_critical_nodes()

        print ("redundant nodes ration: {} time took to process: {} seconds".format(
            float(len(redundant_nodes)) / len(G.nodes()),
            tools.get_time_difference_seconds(start_time_2)
        ))
        print ("------------------------------------------")


class GeneralTools:
    @staticmethod
    def get_percentage_redundant_nodes(networkx_digraph, show_plots=False, network_id=1):
        ex = Experiment(debug=False, draw_graphs=False)
        start_time_2 = datetime.now()

        if show_plots:
            count, bins, ignored = plt.hist(networkx_digraph.in_degree().values(), 50, normed=True)
            plt.title("in-deg from graph")
            # plt.show()
            count, bins, ignored = plt.hist(networkx_digraph.out_degree().values(), 50, normed=True, color='orange')
            plt.title("out-deg from graph")
            plt.show()

        print ("number of self loops: " + str(networkx_digraph.number_of_selfloops()))
        networkx_digraph.remove_edges_from(networkx_digraph.selfloop_edges())

        network_cn = ex.networkx_to_snap_cnetwork(networkx=networkx_digraph, name="", network_id=network_id,
                                                  model=constants.NetworkModel.real_network())

        unmatched_nodes_inset, mds = network_cn.control.snap_find_mds_minimum_driver_node_set()
        redundant_nodes, intermittent_nodes, critical_nodes = \
            network_cn.control.snap_find_redundant_intermittent_critical_nodes()

        perc_r = round(float(len(redundant_nodes)) / len(networkx_digraph.nodes()), 3)
        perc_c = round(float(len(critical_nodes)) / len(networkx_digraph.nodes()), 3)

        print ("   -->N-r: **{} N-c: {} time took to process: {} seconds".format(
            perc_r,
            perc_c,
            tools.get_time_difference_seconds(start_time_2)
        ))

        return perc_r

    @staticmethod
    def copy_graphs_with_respecting_to_redundant_percentage():
        from_path = "D:\\Temp\\random_graph"
        to_path = "D:\\Temp\\low_r"

        for (dirpath, dirnames, filenames) in walk(from_path):
            if dirpath == from_path:
                for filename in filenames:
                    # for undirected random graphs: r_0.1_k_9.75375_n_800_l_7803_p_0.0248_DegreeVariance_18.13994375
                    spl = filename.split('_')

                    if float(spl[1]) < 0.3:
                        copyfile(dirpath + '\\' + filename,
                                 to_path + '\\' + ("{0:.4f}".format(float(spl[1]))).replace('.', '.') + filename)

    @staticmethod
    def generate_random_network_id():
        file_names = tools.directory_files(constants.path_bipartite_representations)

        while True:
            network_id = int(random.uniform(400, 2000))
            if len([tmp for tmp in file_names if tmp.startswith(str(network_id))]) == 0:
                break

        return network_id


class RandomGraphs:
    @staticmethod
    def generate_random_graph(n, p, directed, network_id, path):
        G = nx.gnp_random_graph(n, p, directed=directed)

        num_nodes = len(G.nodes())
        num_links = len(G.edges())
        k_avg = num_links / float(num_nodes)
        print (
            "number of nodes: %i  link#: %i <k>: %f" % (num_nodes, num_links, round(k_avg, 2)))

        if directed:
            print(
                "calc_p: {} [OUT variance: {}] [in variance: {}] mean_degree: {}".format(
                    k_avg / (num_nodes - 1),
                    np.var(G.out_degree().values()),
                    np.var(G.in_degree().values()),
                    np.mean(G.in_degree().values())
                ))
        else:
            print(
                "actual_p: {} calculated_p: {} [degree variance: {} mean: {}]".format(
                    p,
                    2 * (k_avg / (num_nodes - 1)),
                    np.var(G.degree().values()),
                    np.mean(G.degree().values()),
                ))

        perc_r = GeneralTools.get_percentage_redundant_nodes(G, show_plots=False, network_id=network_id)

        if directed:
            output_file_name = "r_{0:.4f}_k_{1:09.4f}_n_{2:06d}_l_{3:010d}_p_{4:07.4f}_InVariance_{5:09.4f}_OutVariance_{6:09.4f}".format(
                perc_r, k_avg, num_nodes, num_links, p, np.var(G.in_degree().values()), np.var(G.out_degree().values()))
        else:
            output_file_name = "r_{0:.4f}_k_{1:09.4f}_n_{2:06d}_l_{3:010d}_p_{4:07.4f}_DegreeVariance_{5:09.4f}".format(
                perc_r, k_avg, num_nodes, num_links, p, np.var(G.degree().values()))

        nx.write_gml(G, path + output_file_name + ".gml")

        return G, perc_r, output_file_name + ".gml"

    @staticmethod
    def experiment(network_id):
        n = 200
        p = 0.005
        r_values = []
        path = "d:\\temp\\random_graph\\n_{0}\\".format(n)
        if not os.path.exists(path):
            os.makedirs(path)
        clipboard_path = "random_graph\\\\{}\\\\".format(n)

        for i in range(1, 17):
            G, perc_r, output_file_name = RandomGraphs.generate_random_graph(
                n, p, False, network_id=network_id, path=path)

            tools.clipboard_copy(clipboard_path + output_file_name)

            # n += 150
            p += 0.0022

            r_values.append(perc_r)

            print ("----------------------------------------------")

        r_values.sort()
        print ("-------------------------------\nMin r is: {}".format(r_values))

    @staticmethod
    def repeat_experiment():
        network_id = GeneralTools.generate_random_network_id()
        for i in range(0, 1000):
            RandomGraphs.experiment(network_id)


if __name__ == '__main__':
    start_time = datetime.now()
    print ("started: " + str(start_time) + "\n;;;;;;;;;;;;;;")

    RandomGraphs.repeat_experiment()

    # GeneralTools.copy_graphs_with_respecting_to_redundant_percentage()

    # load_network_from_text(path="d:\\temp\\g.txt")

    # random_network_given_degree_sequence()

    # experiment_scale_free_hidden_parameter_from_network_science_book(show_plots=True)
    # experiment_scale_free_from_bimodal_paper(show_plots=False)

    # nx.write_gml(G, "d:\\temp\\test2.gml")

    # # tools.networkx_draw(G, "d:\\temp\\scale_free_bimodal.png")
    # wtf()

    # bfs()

    ############# These two lines complement each other
    # find_control_nodes()
    # find_redundant_nodes(initialize=False)

    # developing_netlogo_communication()
    # networkx_gml_test()

    # network_degree_distribution_from_path("d:\\temp\\Centralized rnd100_0.07_test_Nr_0.58.gml")
    # network_degree_distribution_from_path("d:\\temp\\Centralized rnd100_0.07_test_Nr_0.58.gml")
    # network_degree_distribution_from_path("d:\\temp\\Distributed rnd100_0.07_test_Nr_0.23.gml")

    # wtf()

    if True:
        print (";;;;;;;start_time: %s end time: %s duration: %s" %
               (
                   str(start_time),
                   str(datetime.now()),
                   str(tools.get_time_difference_seconds(start_time, datetime.now()))
               ))
