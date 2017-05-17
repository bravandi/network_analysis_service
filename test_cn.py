import random
import pprint
from Tkinter import Tk
from complex_networks.network import Network
from complex_networks.experiment import Experiment
from complex_networks.general_tools import GeneralTools
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
import operator
from complex_networks.tools import networkx_draw


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

        is_binary_file=False,
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


def load_network_from_text(path):
    ex = Experiment(debug=False, draw_graphs=False)

    network_cn = ex.snap_load_network(

        graph_path=path,
        network_id=GeneralTools.generate_random_network_id(),
        name='small_sample',

        is_binary_file=False,
        initialize_graph=True,
        model=constants.NetworkModel.real_network(),

    )

    # print("out var: {} in var: {}".format(str(np.var(G.out_degree().values())), np.var(G.in_degree().values())))

    unmatched_nodes_inset, mds = network_cn.control.snap_find_mds_minimum_driver_node_set()
    redundant_nodes, intermittent_nodes, critical_nodes, augmenting_path_list, other_output = \
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

    print ("nd: {} nr: {}".format(
        len(unmatched_nodes_inset),
        float(len(redundant_nodes)) / network_cn.graph.GetNodes())
    )


class ScaleFree:
    @staticmethod
    def scale_free_gen():
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
        print (
            "number of nodes: %i  link#: %i <k>: %f" % (num_nodes, num_links, round(num_links / float(num_nodes), 2)))
        # nx.draw(G2)

        pos = nx.spring_layout(G2)
        f1 = plt.figure(3, figsize=(12, 12))
        nx.draw_networkx_nodes(G2, pos, cmap=plt.get_cmap('jet'))
        # nx.draw_networkx_edges(G2, pos, edgelist=red_edges, edge_color='r', arrows=True)
        # nx.draw_networkx_edges(G2, pos, edgelist=black_edges, arrows=False)
        nx.draw_networkx_edges(G2, pos, edgelist=G2.edges(), arrows=True)

        print ([nid for nid, val in G2.degree().items() if val == 0])

        # plt.show()

    @staticmethod
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

    @staticmethod
    def experiment_scale_free_hidden_parameter_from_network_science_book(show_plots=True):
        ex = Experiment(debug=False, draw_graphs=False)

        for i in range(0, 11):
            start_time_2 = datetime.now()

            G = ScaleFree.scale_free_hidden_parameter_from_network_science_book(
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
            redundant_nodes, intermittent_nodes, critical_nodes, augmenting_path_list, other_output = \
                network_cn.control.snap_find_redundant_intermittent_critical_nodes()

            print ("@@redundant nodes ratio: {} <k>: {} time took to process: {} seconds".format(
                float(len(redundant_nodes)) / len(G.nodes()),
                round(k_avg, 2),
                tools.get_time_difference_seconds(start_time_2)
            ))
            print ("------------------------------------------")

    @staticmethod
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
        print (
            "number of nodes: %i  link#: %i <k>: %f" % (num_nodes, num_links, round(num_links / float(num_nodes), 2)))

        output_file_name = "alphaOut_{0}_alphaIn_{1}_k_{2}_n_{3}_e_{4}".format(
            round(lambda_out, 3), round(lambda_in, 3), k, num_nodes, num_links)

        tools.clipboard_copy("scale_free_bimodal\\\\" + output_file_name)

        nx.write_gml(G, "d:\\temp\\scale_free_bimodal\\" + output_file_name + ".gml")

        print ("       d:\\temp\\scale_free_bimodal\\" + output_file_name + ".gml")

        return G

    @staticmethod
    def experiment_scale_free_from_bimodal_paper(show_plots=True):
        load_network_from_text(path="d:\\temp\\g.txt")

        ex = Experiment(debug=False, draw_graphs=False)

        p = 0

        for k_avg in range(6, 10):
            start_time_2 = datetime.now()

            G = ScaleFree.scale_free_from_bimodal_paper(
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
                "      out var: {} in var: {}".format(str(np.var(G.out_degree().values())),
                                                      np.var(G.in_degree().values())))

            network_cn = ex.networkx_to_snap_cnetwork(networkx=G, name="", network_id=100 + k_avg,
                                                      model=constants.NetworkModel.real_network())

            unmatched_nodes_inset, mds = network_cn.control.snap_find_mds_minimum_driver_node_set()
            redundant_nodes, intermittent_nodes, critical_nodes, augmenting_path_list, other_output = \
                network_cn.control.snap_find_redundant_intermittent_critical_nodes()

            print ("@@redundant nodes ratio: {} <k>: {} time took to process: {} seconds".format(
                float(len(redundant_nodes)) / len(G.nodes()),
                round(k_avg, 2),
                tools.get_time_difference_seconds(start_time_2)
            ))
            print ("------------------------------------------")


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

        found_a_matchinf_yields_another_r_d = False
        perc_compare = []
        for j in range(0, 1):
            perc_r = GeneralTools.identify_node_types(G, show_plots=False, network_id=network_id)
            if len(perc_compare) > 0 and perc_r not in perc_compare:
                found_a_matchinf_yields_another_r_d = True
                break
            perc_compare.append(perc_r)

        if directed:
            output_file_name = "r_{0:.4f}_k_{1:09.4f}_n_{2:06d}_l_{3:010d}_p_{4:07.4f}_InVariance_{5:09.4f}_OutVariance_{6:09.4f}".format(
                perc_r, k_avg, num_nodes, num_links, p, np.var(G.in_degree().values()), np.var(G.out_degree().values()))
        else:
            output_file_name = "r_{0:.4f}_k_{1:09.4f}_n_{2:06d}_l_{3:010d}_p_{4:07.4f}_DegreeVariance_{5:09.4f}".format(
                perc_r, k_avg, num_nodes, num_links, p, np.var(G.degree().values()))

        if found_a_matchinf_yields_another_r_d:
            nx.write_gml(G, "D:\\Temp\\random_graph\\test\\" + output_file_name + ".gml")

        nx.write_gml(G, path + output_file_name + ".gml")

        return G, perc_r, output_file_name + ".gml"

    @staticmethod
    def experiment(network_id, n=11, p=0.4, p_plus_plus=0.1):

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
            p += p_plus_plus
            if p > 1:
                break

            r_values.append(perc_r)

            print ("----------------------------------------------")

        r_values.sort()
        print ("-------------------------------\nMin r is: {}".format(r_values))

    @staticmethod
    def experiment_switch_link_direction(
            input_networkx_digraph, root_folder_work, draw_graphs=True, network_id=None,
            draw_bipartite_matching_for_each_link_switch=True, analise_all_links=False
    ):
        if network_id is None:
            network_id = GeneralTools.generate_random_network_id()

        perc_r_orig, redundant_nodes_orig, intermittent_nodes_orig, critical_nodes_orig, mds_orig, other_output_orig = \
            GeneralTools.identify_node_types(
                networkx_digraph=input_networkx_digraph, debug=False, draw_graphs=draw_graphs, show_plots=False,
                network_id=network_id, root_folder_work=root_folder_work, draw_label_prefix="INPUT-GRAPH."
            )

        network_cn_orig = other_output_orig['network_cn']
        ex_orig = other_output_orig['experiment']
        augmenting_path_list_orig = other_output_orig['augmenting_path_list']

        n = len(input_networkx_digraph.nodes())

        networkx_bipartite_representation_orgi = ex_orig.snap_to_networkx_cnetwork(
            snap_g=other_output_orig['bipartite_representation_tungraph'], name=network_cn_orig.name,
            network_id=network_cn_orig.network_id, model=network_cn_orig.model, graph_type='undirected').graph
        n_bipartite_representation_orgi = len(networkx_bipartite_representation_orgi.nodes())

        original_graph_control_properties = {
            "00 Nr_perc": perc_r_orig,
            "01 Nr": len(redundant_nodes_orig),
            "02 Ni": len(intermittent_nodes_orig),
            "03 Nc": len(critical_nodes_orig),
            "04 Nd_number_unmatched": len(mds_orig),
            "05 |M|": len(input_networkx_digraph.nodes()) - len(mds_orig),
            "06 n": n,
            "07 #links": len(input_networkx_digraph.edges())
        }

        if draw_bipartite_matching_for_each_link_switch:
            label = "Do -/+{2} to get outset/inset. Unmatched({6}): {0}\nMatched({7}): {1} \nRedundant({8}):{3} Intermittent({9}): {4} Critical({10}): {5}".format(
                ["{}|{}".format(node, node + n + 1) for node in mds_orig],
                ["{}|{}".format(node, node + n + 1) for node in
                 set(input_networkx_digraph.nodes()) - set(other_output_orig['unmatched_nodes'])],
                n + 1, redundant_nodes_orig, intermittent_nodes_orig, critical_nodes_orig,
                len(mds_orig), len(input_networkx_digraph.nodes()) - len(mds_orig), len(redundant_nodes_orig),
                len(intermittent_nodes_orig),
                len(critical_nodes_orig)
            )

            # label = "Unmatched: {0} Matched: {1} Do -/+{2} to get outset/inset\nRedundant:{3} Intermittent: {4} Critical: {5}".format(
            #     ["{}|{}".format(node, node + n + 1) for node in mds_orig],
            #     ["{}|{}".format(node, node + n + 1) for node in
            #      set(input_networkx_graph.nodes()) - set(other_output_orig['unmatched_nodes'])],
            #     n + 1, redundant_nodes_orig, intermittent_nodes_orig, critical_nodes_orig
            # )

            # GeneralTools.draw_bipartite_rep_graph(
            #     ex=ex_orig,
            #     networkx_graph=networkx_bipartite_representation_orgi,
            #     original_graph_nodes=input_networkx_graph.nodes(),
            #     graph_type='undirected',
            #     file_name='{0:04d}_Nr_{1:07.4f}_{2}'.format(0, perc_r_orig, 'bipartite_red_colored'),
            #     network_cn=network_cn_orig, augmenting_path_list=augmenting_path_list_orig, label_draw=label)

            GeneralTools.draw_bipartite_rep_graph(
                ex=ex_orig,
                networkx_graph=networkx_bipartite_representation_orgi,
                original_graph_nodes=input_networkx_digraph.nodes(),
                graph_type='multigraph',
                file_name='{0:04d}_Nr_{1:07.4f}_{2}'.format(0, perc_r_orig, 'bipartite_red_multigraph_color'),
                network_cn=network_cn_orig, augmenting_path_list=augmenting_path_list_orig, label_draw=label)

        edges_betweenness_centrality_orgi = nx.edge_betweenness_centrality(networkx_bipartite_representation_orgi)
        tmp = {}
        for key, value in edges_betweenness_centrality_orgi.items():
            tmp[key] = round(value, 2)
        edges_betweenness_centrality_orgi = tmp
        del tmp

        nodes_betweenness_centrality_orgi = nx.betweenness_centrality(networkx_bipartite_representation_orgi)
        nodes_latapy_clustering = nx.bipartite.latapy_clustering(
            networkx_bipartite_representation_orgi,
            networkx_bipartite_representation_orgi.nodes())
        nodes_closeness_centrality = nx.bipartite.closeness_centrality(
            networkx_bipartite_representation_orgi,
            networkx_bipartite_representation_orgi.nodes())

        n = len(input_networkx_digraph.nodes())
        in_set_cutoff = max(input_networkx_digraph.nodes())
        i = 1

        if analise_all_links is True:
            stats = []
            stats2 = {}

            for edge in input_networkx_digraph.edges():
                # if (edge[0] == 7 and edge[1] == 21) is False:
                #     # print ("x")
                #     continue

                # todo figure out why if dont copy it wont work right ......
                input_networkx_graph_2 = nx.DiGraph(input_networkx_digraph)

                input_networkx_graph_2.remove_edge(edge[0], edge[1])
                print ("#{} | switch {} - {} TO {} - {}".format(i, edge[0], edge[1], edge[1], edge[0]))

                input_networkx_graph_2.add_edge(edge[1], edge[0])

                perc_r, redundant_nodes, intermittent_nodes, critical_nodes, mds, other_output = \
                    GeneralTools.identify_node_types(
                        networkx_digraph=input_networkx_graph_2, debug=False, draw_graphs=draw_graphs,
                        show_plots=False, network_id=network_id,
                        root_folder_work="{}\\{}-{}".format(root_folder_work, edge[0], edge[1]),
                        draw_label_prefix="SWITCHED LINK."
                    )
                network_cn = other_output['network_cn']
                ex = other_output['experiment']
                augmenting_path_list = other_output['augmenting_path_list']

                if draw_bipartite_matching_for_each_link_switch:
                    def on_before_draw(netx_g):
                        # netx_g.add_edge(edge[0], edge[1] + (n + 1), style='bold')
                        # the key 0 is the default key so it makes the first edge bold
                        for edge_g in netx_g.edges():
                            netx_g.add_edge(edge_g[0], edge_g[1], 0, style='bold')

                        netx_g.add_edge(edge[1], edge[0] + (n + 1), 0, style='dashed')

                    root_folder_work_before_change = ex.root_folder_work
                    # ex.root_folder_work = ex.root_folder_work.split('\\')[0] + "\\switched_links"
                    ex.root_folder_work = tools.path_split(ex.root_folder_work, 0)[1] + "\\switched_links"

                    label = "Do -/+{2} to get outset/inset. Unmatched({6}): {0}\nMatched({7}): {1} \nRedundant({8}):{3} Intermittent({9}): {4} Critical({10}): {5}".format(
                        ["{}|{}".format(node, node + n + 1) for node in mds],
                        ["{}|{}".format(node, node + n + 1) for node in
                         set(input_networkx_graph_2.nodes()) - set(other_output['unmatched_nodes'])],
                        n + 1, redundant_nodes, intermittent_nodes, critical_nodes,
                        len(mds), len(input_networkx_graph_2.nodes()) - len(mds), len(redundant_nodes),
                        len(intermittent_nodes),
                        len(critical_nodes)
                    )

                    saved_pic_path = GeneralTools.draw_bipartite_rep_graph(
                        ex=ex,
                        snap_graph=other_output['bipartite_representation_tungraph'],
                        file_name="{0:04d}_Nr_{1:07.4f}_switchedEdge_{2}-{3}_to_{3}-{2}_({4}-{5}_to_{6}-{7})".format(
                            i, perc_r,
                            edge[0], edge[1],
                            edge[0], edge[1] + (n + 1),
                            edge[1], edge[0] + (n + 1)),
                        original_graph_nodes=input_networkx_graph_2.nodes(), graph_type='multigraph',
                        network_cn=network_cn, augmenting_path_list=augmenting_path_list,
                        on_before_draw=on_before_draw, label_draw=label
                    )
                    # need lock here | sometime os is slow to create the file
                    while os.path.isfile(saved_pic_path) is False:
                        pass
                    copyfile(
                        saved_pic_path,
                        os.path.join(constants.path_draw_graphs, root_folder_work_before_change,
                                     tools.path_split(saved_pic_path, -1)[1]))

                # input_networkx_graph_2.add_edge(edge[0], edge[1], color='blue')

                # ;;;;;;;;;;;;;;;; start analyses on the bipartite representation of the input graph ;;;;;;;;;;;;;;;

                bipartite_representation_orgi_total_degree = networkx_bipartite_representation_orgi.degree()
                # input_graph_in_degree = networkx_bipartite_representation_orgi.in_degree()
                # input_graph_out_degree = networkx_bipartite_representation_orgi.out_degree()
                in_set_node_id = edge[1] + n + 1
                tmp = \
                    {
                        '00 Nr_perc': perc_r,
                        '00 key': "({0},{1})->({0},{2})".format(edge[0], edge[1], in_set_node_id),
                        "01 Nr": len(redundant_nodes),
                        "02 Ni": len(intermittent_nodes),
                        "03 Nc": len(critical_nodes),
                        "04 Nd_number_unmatched": len(mds),
                        # 'Ni': other_output['perc_i'],
                        # 'Nc': other_output['perc_c'],
                        '05 |M|': len(input_networkx_graph_2) - len(mds),

                        '06 Edge Betweenness Centrality': edges_betweenness_centrality_orgi[(edge[0], in_set_node_id)],
                        '07 From Node Betweenness Centrality': nodes_betweenness_centrality_orgi[edge[0]],
                        '08 To Node Betweenness Centrality': nodes_betweenness_centrality_orgi[in_set_node_id],

                        '09 From Node Closeness Centrality': nodes_closeness_centrality[edge[0]],
                        '10 To Node Closeness Centrality': nodes_closeness_centrality[in_set_node_id],

                        '11 From Node Latapy Clustering': nodes_latapy_clustering[edge[0]],
                        '12 To Node Latapy Clustering': nodes_latapy_clustering[in_set_node_id],

                        '13 From node degrees': "total: {}".format(bipartite_representation_orgi_total_degree[edge[0]]),
                        '14 To node degrees': "total: {}".format(
                            bipartite_representation_orgi_total_degree[in_set_node_id])
                    }

                stats.append(tmp)
                stats2[(edge[0], edge[1])] = tmp
                stats2[(edge[0], in_set_node_id)] = tmp
                i += 1

            stats.sort(key=lambda x: x["00 Nr_perc"], reverse=True)

        output = "\tOriginal graph control properties:\n{}\n" \
            .format(pprint.pformat(original_graph_control_properties))

        output += "\n;;;;;;;;;;;;;;;Augmenting Paths;;;;;;;;;;;;;;;;;\n"
        augmenting_path_list_orig_list = []
        augmenting_links = set()

        def convert_node_id_to_ordinal_id(node_id, cut_off, number_of_nodes):
            if node_id > cut_off:
                return node_id - (number_of_nodes + 1)
            return node_id

        def convert_list_link_to_original_id(links):
            result = ""
            if type(links) is not list:
                links = [links]
            for item in links:
                result += "({},{}->{},{}), ".format(item[0], item[1],
                                                    convert_node_id_to_ordinal_id(item[0], in_set_cutoff, n),
                                                    convert_node_id_to_ordinal_id(item[1], in_set_cutoff, n))
            return result[:-1]

        for index in range(0, len(augmenting_path_list_orig)):
            tmp = []
            for index_j in range(0, len(augmenting_path_list_orig[index]) - 1):
                a = augmenting_path_list_orig[index][index_j]
                b = augmenting_path_list_orig[index][index_j + 1]
                if a > b:
                    link = (b, a)
                else:
                    link = (a, b)

                tmp.append(link)
                augmenting_links.add(link)

            augmenting_path_list_orig_list.append((
                # (path size, path)
                len(tmp), tmp)
            )

        augmenting_path_list_orig_list.sort(key=lambda x: x[0], reverse=True)

        for item in augmenting_path_list_orig_list:
            output += "\tSize {}: {}\n".format(item[0], convert_list_link_to_original_id(item[1]))

        output += "Frequency:\n"
        link_frequency = []
        for link in augmenting_links:
            freq = 0
            for path in augmenting_path_list_orig_list:
                for link_path in path[1]:
                    if link_path == link:
                        freq += 1
            link_frequency.append(
                # (frequency, link)
                (freq, link)
            )

        link_frequency.sort(key=lambda x: x[0], reverse=True)

        for item in link_frequency:
            if analise_all_links is True:
                tmp = stats2[item[1]]['00 Nr_perc']

            output += "\tFreq {}: {} Switched to ({}, {}) Nr_perc: {}\n".format(
                item[0], convert_list_link_to_original_id(item[1]),
                convert_node_id_to_ordinal_id(item[1][1], in_set_cutoff, n), item[1][0] + n + 1,
                tmp
            )

        # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;GRAPH ALGORITHMS;;;;;;;;;;;;;;;;;;;;;;;;;;;
        def normalize_dictionary(dict, in_set_cutoff, main_graph):
            result = {}
            n = len(main_graph.nodes())
            result[n] = "---INSET:---"
            for out_set_node_id in main_graph.nodes():

                in_set_node_id = out_set_node_id + n + 1

                prefix = ""
                if out_set_node_id in dict:
                    result[out_set_node_id] = prefix + "{0:.2f}".format(round(dict[out_set_node_id], 3))
                else:
                    result[out_set_node_id] = prefix + "NaN"

                prefix = "{}|".format(out_set_node_id)
                if in_set_node_id in dict:
                    result[in_set_node_id] = prefix + "{0:.2f}".format(round(dict[in_set_node_id], 3))
                else:
                    result[in_set_node_id] = prefix + "NaN"

            return result

        output += "\n--------------------Graph Algorithms-------------------------" \
                  "\n\t(starting index {0} is the InSet. Do -{1} to get original node id. " \
                  "\n\tThere is no node {0} in the bipartite representation of the graph):\n\n". \
            format(in_set_cutoff, n + 1)

        output += "Edges Betweenness Centrality: {}\n".format(edges_betweenness_centrality_orgi)
        output += "\nNodes Betweenness Centrality: {}\n".format(normalize_dictionary(
            nodes_betweenness_centrality_orgi, in_set_cutoff, input_networkx_digraph))
        output += "\nNodes Latapy Clustering     : {}\n".format(normalize_dictionary(
            nodes_latapy_clustering, in_set_cutoff, input_networkx_digraph))
        output += "\nNodes Closeness Centrality  : {}\n".format(normalize_dictionary(
            nodes_closeness_centrality, in_set_cutoff, input_networkx_digraph))
        # ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;GRAPH ALGORITHMS;;;;;;;;;;;;;;;;;;;;;;;;;;;

        if analise_all_links is True:
            output += "\n\n---------------------------Partitions-----------------------------\n"

            cut_of_partition = []
            cut_of_value = 0.7
            other_nodes = []
            for value in stats:
                if value["00 Nr_perc"] > cut_of_value:
                    cut_of_partition.append(value)
                else:
                    other_nodes.append(value)

            def calc_mean(x, key):
                # try:
                #     s = sum([nc[key] for nc in x])
                # except Exception:
                #     return ("{0} centrality mean: {1}\n".format(key, "None"))
                try:
                    s = 0
                    for item in x:
                        try:
                            s += item[key]
                        except Exception:
                            pass

                    return ("{0} centrality mean: {1}\n".format(key, s / float(len(x))))
                except Exception:
                    return ("{0} centrality mean: {1}\n".format(key, "None"))

            def calc_overall_stats(dict_values):
                result = ""
                result += calc_mean(dict_values, '06 Edge Betweenness Centrality')
                result += calc_mean(dict_values, '07 From Node Betweenness Centrality')
                result += calc_mean(dict_values, '08 To Node Betweenness Centrality')
                result += calc_mean(dict_values, '09 From Node Closeness Centrality')
                result += calc_mean(dict_values, '10 To Node Closeness Centrality')
                result += calc_mean(dict_values, '11 From Node Latapy Clustering')
                result += calc_mean(dict_values, '12 To Node Latapy Clustering')
                return result

            # output += "\n\tParition Nodes Nr > {}\n".format(cut_of_value)
            # output += calc_overall_stats(cut_of_partition)
            #
            # output += "\n\tOther nodes Nr =< {}\n".format(cut_of_value)
            # output += calc_overall_stats(other_nodes)

            output += "\n\tAll nodes together\n"
            output += calc_overall_stats(stats)

        if analise_all_links is True:
            output += "\n-------------------------Switch Links Analysis----------------------------\n"
            output += "stats switch link results:\n"
            for item in stats:
                output += "__________________________________________________________________________" \
                          "\n{}: {}\n".format(item['00 key'], pprint.pformat(item, indent=10))

        try:
            with open(tools.absolute_path(
                    "{}{}/output.txt".format(constants.path_draw_graphs, root_folder_work)), 'w') as writefile:
                writefile.write(output)
        except Exception:
            pass

        try:
            with open(tools.absolute_path(
                    "{}{}/output.txt".format(constants.path_bipartite_representations, root_folder_work)),
                    'w') as writefile:
                writefile.write(output)
        except Exception:
            pass

        print (output)

    @staticmethod
    def repeat_experiment():
        network_id = GeneralTools.generate_random_network_id()

        start_n = 20
        for i in range(0, 1):
            RandomGraphs.experiment(network_id, n=start_n)
            start_n += 1

    @staticmethod
    def random_network_given_degree_sequence():
        ex = Experiment(debug=False, draw_graphs=False)
        show_plots = True

        deg_seq_length = 500
        for i in range(0, 1):
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
            redundant_nodes, intermittent_nodes, critical_nodes, augmenting_path_list, other_output = \
                network_cn.control.snap_find_redundant_intermittent_critical_nodes()

            print ("redundant nodes ration: {} time took to process: {} seconds".format(
                float(len(redundant_nodes)) / len(G.nodes()),
                tools.get_time_difference_seconds(start_time_2)
            ))
            print ("------------------------------------------")

    @staticmethod
    def load_undirected_convert_to_directed():
        #
        G_u = Network.networkx_create_from_gml(
            "D:\\temp\\random_graph\\n_11\\r_0.0000_k_0004.1818_n_000011_l_0000000046_p_00.8000_DegreeVariance_0001.6860.gml"
        )

        G = nx.DiGraph()
        # for edge in G_u.edges():
        #     G.add_edge(edge)
        G.add_edges_from(G_u.edges())

        # '';;;;;;;;;;;;;;;;;;;;;;;these are for validtion
        # G_from_netlogo = Network.networkx_create_from_gml(
        #     "D:\\temp\\netlogo-diffusion.gml"
        # )
        #
        perc_r, redundant_nodes, intermittent_nodes, critical_nodes, mds, other_output = \
            GeneralTools.identify_node_types(
                networkx_digraph=G, debug=False, draw_graphs=False, show_plots=False, network_id=21
            )
        #
        # a = set()
        # b = set()
        # for i in G.edges():
        #     a.add(i)
        # for i in G_from_netlogo.edges():
        #     b.add(i)

        return G

    @staticmethod
    def experiment_switch_link_direction_controller():
        draw_bipartite_matching_for_each_link_switch = True
        draw_graphs = True
        analise_all_links = True
        switch_links = []

        # path = "D:\\temp\\low_r\\n_5\\0.2000r_0.2000_k_0001.8000_n_000005_l_0000000009_p_00.9000_DegreeVariance_0000.2400.gml"

        # switch_links = [(5, 9), (1, 4), (2, 6)]
        # path = "D:\\temp\\low_r\\n_10\\0.0000r_0.0000_k_0002.4000_n_000010_l_0000000024_p_00.6000_DegreeVariance_0001.9600.gml"

        # path = "D:\\temp\\low_r\\n_5\\0.0000r_0.0000_k_0001.6000_n_000005_l_0000000008_p_00.8000_DegreeVariance_0000.5600.gml"
        # path = "D:\\temp\\low_r\\n_20\\0.2500r_0.2500_k_0005.8500_n_000020_l_0000000117_p_00.6000_DegreeVariance_0003.9100.gml"
        path = "D:\\temp\\low_r\\n_5\\0.2000r_0.2000_k_0001.8000_n_000005_l_0000000009_p_00.9000_DegreeVariance_0000.2400.gml"
        # path = "D:\\temp\\random_graph\\n_10_p_0.4.gml"
        # path = "D:\\SoftwareProject\\network_analysis_service\\data\\Neural Network\\celegansneural.gml"
        # path = 'D:\\Temp\\IMPORTANT_CAUSE ERROR\\netlogo-diffusion_n3_e4.gml'

        path_parts = path.split('\\')
        G = Network.networkx_create_from_gml(
            # path="d:\\temp\\netlogo-diffusion2.gml"
            # path="d:\\temp\\netlogo-diffusion.gml"
            # path="D:\\temp\\low_r\\n_15\\0.1330r_0.1330_k_0006.4000_n_000015_l_0000000096_p_00.9000_DegreeVariance_0001.2267.gml"
            # path="D:\\SoftwareProject\\complex_networks_tools\\data\\Neural Network\\celegansneural.gml"
            path=path
            # path="D:\\temp\\random_graph\\n_11\\r_0.0000_k_0004.1818_n_000011_l_0000000046_p_00.8000_DegreeVariance_0001.6860.gml"
            , label='id',
            convert_to_digraph=True
        )

        root_folder_work = path_parts[len(path_parts) - 1][:-4]
        if len(switch_links) > 0:
            root_folder_work += "_!!!SW_"
            for link in switch_links:
                if G.has_edge(link[0], link[1]) is False:
                    raise Exception("Link does not exists {}".format(link))
                G.remove_edge(link[0], link[1])
                G.add_edge(link[1], link[0])
                root_folder_work += "{}-{}_".format(link[0], link[1])

            root_folder_work = root_folder_work[:-1]
        #
        copy_to = tools.absolute_path(constants.path_bipartite_representations + root_folder_work)
        if not os.path.exists(copy_to):
            os.makedirs(copy_to)
        copy_to += "\\" + tools.path_split(path, -1)[1]
        networkx_gml.write_gml(G, copy_to)
        # copyfile(path, copy_to + "\\" + tools.path_split(path, -1)[1])

        RandomGraphs.experiment_switch_link_direction(
            input_networkx_digraph=G, root_folder_work=root_folder_work, draw_graphs=draw_graphs,
            network_id=None, draw_bipartite_matching_for_each_link_switch=draw_bipartite_matching_for_each_link_switch,
            analise_all_links=analise_all_links
        )


if __name__ == '__main__':
    start_time = datetime.now()
    print ("started: " + str(start_time) + "\n;;;;;;;;;;;;;;")

    # RandomGraphs.experiment_switch_link_direction_controller()

    # GeneralTools.identify_node_type_given_path(
    #     "D:\\Temp\\scale_free_bimodal\\scale_free_k_14_n_1000_lambda_out_2.67_lambda_in_3.0_.gml"
    # )
    path = "D:\\Temp\\scale_free\\"
    # path +=  "n_5000_alpha_0.08_beta_0.5_gamma_0.42_deltaIn_0.0_deltaOut_0.2.gml"
    # path += "n_5000_alpha_0.41_beta_0.54_gamma_0.05_deltaIn_0.2_deltaOut_0.gml"
    # path += "n_10000_alpha_0.08_beta_0.5_gamma_0.42_deltaIn_0.0_deltaOut_0.2.gml"
    path += "n_100000_alpha_0.08_beta_0.5_gamma_0.42_deltaIn_0.0_deltaOut_0.2.gml"

    load_network_from_text(path="d:\\temp\\nr_kav_nm_SF_0.5_0.54_1000.txt")

    # GeneralTools.identify_node_type_given_path(path=path)

    n = 5000
    alpha = 0.08
    beta = 0.50
    gamma = 0.42
    delta_in = 0.0
    delta_out = 0.2

    # default
    alpha = 0.41
    beta = 0.54
    gamma = 0.05
    delta_in = 0.2
    delta_out = 0

    # G = nx.scale_free_graph(n=n, alpha=alpha, beta=beta, gamma=gamma, delta_in=delta_in, delta_out=delta_out)
    #
    # networkx_gml.write_gml(
    #     G,
    #     "d:\\temp\\scale_free\\n_{}_alpha_{}_beta_{}_gamma_{}_deltaIn_{}_deltaOut_{}.gml".format(
    #         n, alpha, beta, gamma, delta_in, delta_out
    #     ))

    # RandomGraphs.repeat_experiment()

    # GeneralTools.identify_node_types(networkx_digraph=G, debug=True, draw_graphs=True, show_plots=False, network_id=1)

    # RandomGraphs.load_undirected_convert_to_directed()
    # GeneralTools.gml_stats()
    # GeneralTools.copy_graphs_with_respecting_to_redundant_percentage()

    # GeneralTools.filter_and_copy_generated_graphs()

    # load_network_from_text(path="d:\\temp\\g.txt")

    # RandomGraphs.random_network_given_degree_sequence()

    # experiment_scale_free_hidden_parameter_from_network_science_book(show_plots=True)
    # experiment_scale_free_from_bimodal_paper(show_plots=False)

    # G = nx.fast_gnp_random_graph(10, 0.4, directed=True)
    # nx.write_gml(G, "d:\\temp\\test2.gml")

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
