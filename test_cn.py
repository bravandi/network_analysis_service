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


def load_network_from_text(path):
    ex = Experiment(debug=False, draw_graphs=False)

    network_cn = ex.snap_load_network(

        graph_path=path,
        network_id=GeneralTools.generate_random_network_id(),
        name='small_sample',

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

    print (float(len(redundant_nodes)) / network_cn.graph.GetNodes())


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


class GeneralTools:
    @staticmethod
    def identify_node_types(
            networkx_digraph, root_folder_work, debug=False, draw_graphs=False, show_plots=False,
            network_id=1):
        ex = Experiment(debug=debug, draw_graphs=draw_graphs, root_folder_work=root_folder_work)
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

        # TODO CREATE A WAY TO CUSTOMIZE SAVE LOCATION NOT JUST BASED ON NETWORK ID
        unmatched_nodes_inset, mds = network_cn.control.snap_find_mds_minimum_driver_node_set()

        redundant_nodes, intermittent_nodes, critical_nodes, augmenting_path_list, other_output = \
            network_cn.control.snap_find_redundant_intermittent_critical_nodes()

        perc_r = round(float(len(redundant_nodes)) / len(networkx_digraph.nodes()), 3)
        perc_c = round(float(len(critical_nodes)) / len(networkx_digraph.nodes()), 3)

        print ("   -->N-r: **{} N-c: {} time took to process: {} seconds".format(
            perc_r,
            perc_c,
            tools.get_time_difference_seconds(start_time_2)
        ))

        other_output['experiment'] = ex
        other_output['network_cn'] = network_cn
        other_output['augmenting_path_list'] = augmenting_path_list

        if draw_graphs is True:
            GeneralTools.draw_bipartite_rep_graph(
                ex=ex, snap_graph=other_output['bipartite_representation_tungraph'], graph_type='undirected',
                file_name="bipartite_red_colored", network_cn=network_cn, augmenting_path_list=augmenting_path_list)

        return perc_r, redundant_nodes, intermittent_nodes, critical_nodes, mds, other_output

    @staticmethod
    def draw_bipartite_rep_graph(ex, network_cn, augmenting_path_list, file_name, graph_type='undirected',
                                 on_before_draw=None, networkx_graph=None, snap_graph=None):

        if snap_graph is not None:
            networkx_graph = ex.snap_to_networkx_cnetwork(
                snap_g=snap_graph, name=network_cn.name,
                network_id=network_cn.network_id, model=network_cn.model, graph_type=graph_type).graph

        # for augmenting_p in augmenting_path_list:
        #     prev_node = -1
        #     color = colors.next()
        #     for node_id in augmenting_p:
        #         if prev_node > -1:
        #             if network_x_bip_rep.graph.has_edge(prev_node, node_id) is False:
        #                 raise Exception("edge does not exists {} - {}".format(prev_node, node_id))
        #             network_x_bip_rep.graph.add_edge(prev_node, node_id, color=color)
        #             prev_node = node_id
        #         else:
        #             prev_node = node_id

        # tools.networkx_draw(
        #     G=networkx_graph,
        #     path="%s/%s/%s.jpg" % (constants.path_draw_graphs, network_cn.network_id, "bipartite_rep_multi_edge"))

        # network_x_bip_rep = network_cn.experiment.snap_to_networkx_cnetwork(
        #     snap_g=snap_graph, name=network_cn.name,
        #     network_id=network_cn.network_id, model=network_cn.model, graph_type='undirected')

        colors = None
        is_multi_graph = False
        if "multi" in graph_type:
            colors = tools.color_generator(len(augmenting_path_list))
            is_multi_graph = True

        color = 'red'

        for augmenting_p in augmenting_path_list:
            prev_node = -1

            if is_multi_graph:
                color = colors.next()

            for node_id in augmenting_p:
                if prev_node > -1:
                    if networkx_graph.has_edge(prev_node, node_id) is False:
                        raise Exception("edge does not exists {} - {}".format(prev_node, node_id))
                    networkx_graph.add_edge(prev_node, node_id, color=color)
                    prev_node = node_id
                else:
                    prev_node = node_id

        if on_before_draw is not None:
            on_before_draw(networkx_graph)

        tools.networkx_draw(
            G=networkx_graph,
            # path="%s/%s/%s.jpg" % (constants.path_draw_graphs, network_cn.network_id, file_name)
            path="%s/%s/%s.jpg" % (constants.path_draw_graphs, ex.root_folder_work, file_name)
        )

        return networkx_graph

    @staticmethod
    def copy_graphs_with_respecting_to_redundant_percentage(from_path, to_path):

        if not os.path.exists(to_path):
            os.makedirs(to_path)

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

    @staticmethod
    def networkx_gml_test():
        # # G = nx.scale_free_graph(200, delta_in=1, delta_out=0, alpha=0.05, beta=0.45, gamma=0.5)
        # # G = nx.grid_2d_graph(10, 10)
        # # G = nx.cycle_graph(50)
        # G = networkx_gml.read_gml("d:\\temp\\netlogo-diffusion.gml", label='id')
        # # G = nx.fast_gnp_random_graph(100, 0.3)
        #
        # # G = nx.gnp_random_graph(100, 0.02, directed=True)
        # # G = nx.gnm_random_graph(100, 200, directed=True)
        # # G = nx.binomial_graph(100, 0.01, directed=True)
        # # in_degree, out_degree, total_degree = Network.nx_degree_distribution(G)
        #
        # # G = graph_generator.gnm_random_graph(200, 400, directed=True)
        # # G = nx.erdos_renyi_graph(200, 0.005, directed=True)
        # # Network.nx_reverse_links(G)
        #
        # # G = nx.gn_graph(50)
        #
        # # G.remove_edges_from(G.selfloop_edges())

        # networkx_gml.write_gml(G, "d:\\temp\\netlogo-diffusion.gml")

        G = Network.networkx_create_from_gml(
            path="d:\\temp\\netlogo-diffusion_0_2_11.gml"
            # path="D:\\SoftwareProject\\complex_networks_tools\\data\\Neural Network\\celegansneural.gml"
        )
        print G.nodes()

        perc_r, redundant_nodes, intermittent_nodes, critical_nodes, mds, other_output = \
            GeneralTools.identify_node_types(networkx_digraph=G, draw_graphs=True)

        mds_who = ""
        for node in mds:
            mds_who += str(dict(G.nodes(True))[node]['WHO']) + ' '

        print (mds_who)

        # G = net.graph
        #
        # num_nodes = len(G.nodes())
        # num_links = len(G.edges())
        # print (
        #     "number of nodes: %i  link#: %i <k>: %f" % (num_nodes, num_links, round(num_links / float(num_nodes), 2)))
        #
        # ex = Experiment(debug=True)
        #
        # cnetwork = ex.networkx_to_snap_cnetwork(
        #     networkx=net,
        #     name="",
        #     model=constants.NetworkModel.netlogo(),
        #     network_id=101
        # )
        #
        # in_degree, out_degree, total_degree = cnetwork.snap_degree_distribution()
        #
        # print (in_degree)
        # print (out_degree)
        #
        # unmatched_nodes_inset, mds = cnetwork.control.snap_find_mds_minimum_driver_node_set()
        #
        # redundant_nodes, intermittent_nodes, critical_nodes = \
        #     cnetwork.control.snap_find_redundant_intermittent_critical_nodes()
        #
        # print("")

    @staticmethod
    def gml_stats():
        G = Network.networkx_create_from_gml(
            path="d:\\temp\\netlogo-diffusion.gml"
            # path="D:\\SoftwareProject\\complex_networks_tools\\data\\Neural Network\\celegansneural.gml"
        )
        G2 = G.to_undirected()

        print(nx.clustering(G2))
        pass


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
            input_networkx_graph, root_folder_work, draw_graphs=True, network_id=None,
            draw_bipartite_matching_for_each_node_switch=True
    ):
        if isinstance(input_networkx_graph, nx.Graph) is True:
            tmp = nx.DiGraph()
            tmp.add_edges_from(input_networkx_graph.edges())

            input_networkx_graph = tmp
            del tmp

        if network_id is None:
            network_id = GeneralTools.generate_random_network_id()
        network_id = 501

        perc_r_orig, redundant_nodes_orig, intermittent_nodes_orig, critical_nodes_orig, mds_orig, other_output_orig = \
            GeneralTools.identify_node_types(
                networkx_digraph=input_networkx_graph, debug=False, draw_graphs=draw_graphs, show_plots=False,
                network_id=network_id, root_folder_work=root_folder_work
            )

        network_cn_orig = other_output_orig['network_cn']
        ex_orig = other_output_orig['experiment']
        augmenting_path_list_orig = other_output_orig['augmenting_path_list']

        n = len(input_networkx_graph.nodes())

        networkx_bipartite_representation_orgi = GeneralTools.draw_bipartite_rep_graph(
            ex=ex_orig, snap_graph=other_output_orig['bipartite_representation_tungraph'], graph_type='undirected',
            file_name='{0:04d}_Nr_{1:07.4f}_{2}'.format(0, perc_r_orig, 'bipartite_rep_colored'),
            network_cn=network_cn_orig, augmenting_path_list=augmenting_path_list_orig)

        GeneralTools.draw_bipartite_rep_graph(
            ex=ex_orig, snap_graph=other_output_orig['bipartite_representation_tungraph'], graph_type='multigraph',
            file_name='{0:04d}_Nr_{1:07.4f}_{2}'.format(0, perc_r_orig, 'bipartite_rep_multigraph_color'),
            network_cn=network_cn_orig, augmenting_path_list=augmenting_path_list_orig)

        edges_centrality = nx.edge_betweenness_centrality(networkx_bipartite_representation_orgi)
        nodes_centrality = nx.betweenness_centrality(networkx_bipartite_representation_orgi)

        n = len(input_networkx_graph.nodes())
        i = 1

        stats = {}

        for edge in input_networkx_graph.edges():
            # if (edge[0] == 7 and edge[1] == 21) is False:
            #     # print ("x")
            #     continue

            # todo figure out why if dont copy it wont work right ......
            input_networkx_graph_2 = nx.DiGraph(input_networkx_graph)

            input_networkx_graph_2.remove_edge(edge[0], edge[1])
            print ("switch {} - {} TO {} - {}".format(edge[0], edge[1], edge[1], edge[0]))

            input_networkx_graph_2.add_edge(edge[1], edge[0], color='green')

            perc_r, redundant_nodes, intermittent_nodes, critical_nodes, mds, other_output = \
                GeneralTools.identify_node_types(
                    networkx_digraph=input_networkx_graph_2, debug=False, draw_graphs=draw_graphs,
                    show_plots=False, network_id=network_id,
                    root_folder_work="{}\\{}-{}".format(root_folder_work, edge[0], edge[1])
                )
            network_cn = other_output['network_cn']
            ex = other_output['experiment']
            augmenting_path_list = other_output['augmenting_path_list']

            def on_before_draw(netx_g):
                netx_g.add_edge(edge[0], edge[1] + (n + 1), color='blue')
                netx_g.add_edge(edge[1], edge[0] + (n + 1), color='green')
                pass

            if draw_bipartite_matching_for_each_node_switch:
                ex.root_folder_work = ex.root_folder_work.split('\\')[0] + "\\switched_links"
                GeneralTools.draw_bipartite_rep_graph(
                    ex=ex, snap_graph=other_output['bipartite_representation_tungraph'],
                    file_name="{0:04d}_Nr_{1:07.4f}_switchedEdge_{2}-{3}_to_{3}-{2}_({4}-{5}_to_{6}-{7})".format(
                        i, perc_r,
                        edge[0], edge[1],
                        edge[0], edge[1] + (n + 1),
                        edge[1], edge[0] + (n + 1)),
                    network_cn=network_cn, augmenting_path_list=augmenting_path_list,
                    on_before_draw=on_before_draw
                )

            input_networkx_graph_2.add_edge(edge[0], edge[1], color='blue')

            stats[(edge[0], edge[1])] = {
                'Nr': perc_r,
                'bipartite_edge_centrality': edges_centrality[(edge[0], edge[1] + (n + 1))],
                'from_centrality': nodes_centrality[edge[0]],
                'to_centrality': nodes_centrality[edge[1] + (n + 1)]
            }

            i += 1

        # print (stats[(8, 10)])
        tmp2 = []
        other_nodes = []
        for item, value in stats.items():
            if value["Nr"] > 0.9:
                tmp2.append(value)
            else:
                other_nodes.append(value)

        def calc_cent(x, key):
            try:
                return ("{0} centrality mean: {1}".format(key, np.mean([nc[key] for nc in x])))
            except Exception:
                return None

        # edges_cent2 = sorted(edges_cent.items(), key=operator.itemgetter(1), reverse=True)
        print (calc_cent(tmp2, 'from_centrality'))
        print (calc_cent(tmp2, 'to_centrality'))
        print (calc_cent(tmp2, 'bipartite_edge_centrality'))
        print (';;;;;;;;;;;;;; not significant ;;;;;;;;;;;;;')
        print (calc_cent(other_nodes, 'from_centrality'))
        print (calc_cent(other_nodes, 'to_centrality'))
        print (calc_cent(other_nodes, 'bipartite_edge_centrality'))
        print (';;;;;;;;;;;;;; All Nodes ;;;;;;;;;;;;;')
        print (calc_cent(stats.values(), 'from_centrality'))
        print (calc_cent(stats.values(), 'to_centrality'))
        print (calc_cent(stats.values(), 'bipartite_edge_centrality'))

        print ("closeness_centrality: {}".format(
            nx.bipartite.closeness_centrality(networkx_bipartite_representation_orgi,
                                              networkx_bipartite_representation_orgi.nodes())))
        print ("latapy_clustering: {}".format(
            nx.bipartite.latapy_clustering(networkx_bipartite_representation_orgi,
                                           networkx_bipartite_representation_orgi.nodes())))

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


if __name__ == '__main__':
    start_time = datetime.now()
    print ("started: " + str(start_time) + "\n;;;;;;;;;;;;;;")
    path = "D:\\temp\\low_r\\n_5\\0.2000r_0.2000_k_0001.8000_n_000005_l_0000000009_p_00.9000_DegreeVariance_0000.2400.gml"

    # RandomGraphs.repeat_experiment()
    G = Network.networkx_create_from_gml(
        # path="d:\\temp\\netlogo-diffusion2.gml"
        # path="d:\\temp\\netlogo-diffusion.gml"
        # path="D:\\temp\\low_r\\n_15\\0.1330r_0.1330_k_0006.4000_n_000015_l_0000000096_p_00.9000_DegreeVariance_0001.2267.gml"
        # path="D:\\SoftwareProject\\complex_networks_tools\\data\\Neural Network\\celegansneural.gml"
        path="D:\\temp\\low_r\\n_5\\0.2000r_0.2000_k_0001.8000_n_000005_l_0000000009_p_00.9000_DegreeVariance_0000.2400.gml"
        # path="D:\\temp\\random_graph\\n_11\\r_0.0000_k_0004.1818_n_000011_l_0000000046_p_00.8000_DegreeVariance_0001.6860.gml"
    )
    # GeneralTools.identify_node_types(networkx_digraph=G, debug=True, draw_graphs=True, show_plots=False, network_id=1)

    RandomGraphs.experiment_switch_link_direction(
        G, "0.2000r_0.2000_k_0001.8000_n_000005_l_0000000009_p_00.9000_DegreeVariance_0000.2400")

    # RandomGraphs.load_undirected_convert_to_directed()
    # GeneralTools.gml_stats()
    # GeneralTools.copy_graphs_with_respecting_to_redundant_percentage()

    # GeneralTools.copy_graphs_with_respecting_to_redundant_percentage(
    #     from_path="D:\\Temp\\random_graph\\n_11",
    #     to_path="D:\\Temp\\low_r\\n_11"
    # )

    # load_network_from_text(path="d:\\temp\\g.txt")

    # random_network_given_degree_sequence()

    # experiment_scale_free_hidden_parameter_from_network_science_book(show_plots=True)
    # experiment_scale_free_from_bimodal_paper(show_plots=False)

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
