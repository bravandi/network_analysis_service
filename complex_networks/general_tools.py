import random
from complex_networks.network import Network
from complex_networks.experiment import Experiment
import complex_networks.constants as constants
import complex_networks.tools as tools
from datetime import datetime
import networkx as nx
import complex_networks.networkx_gml as networkx_gml
import matplotlib.pyplot as plt
import os
from os import walk
from shutil import copyfile


class GeneralTools:
    @staticmethod
    def identify_node_type_given_path(path):
        G = networkx_gml.read_gml(path, label='id')

        GeneralTools.identify_node_types(
            networkx_digraph=G,
            root_folder_work=tools.path_split(path, -1)[1],
            debug=True,
            draw_graphs=False,
            show_plots=False,
            network_id=1,
            draw_label_prefix='')

    @staticmethod
    def identify_driver_nodes(
            networkx_digraph, root_folder_work, debug=False, draw_graphs=False, show_plots=False,
            network_id=1, draw_label_prefix=''):
        ex = Experiment(debug=debug, draw_graphs=draw_graphs, root_folder_work=root_folder_work)
        if show_plots:
            count, bins, ignored = plt.hist(networkx_digraph.in_degree().values(), 50, normed=True)
            plt.title("in-deg from graph")
            # plt.show()
            count, bins, ignored = plt.hist(networkx_digraph.out_degree().values(), 50, normed=True, color='orange')
            plt.title("out-deg from graph")
            plt.show()

        if debug:
            print ("number of self loops: " + str(networkx_digraph.number_of_selfloops()))
            pass

        networkx_digraph.remove_edges_from(networkx_digraph.selfloop_edges())

        network_cn = ex.networkx_to_snap_cnetwork(networkx=networkx_digraph, name="", network_id=network_id,
                                                  model=constants.NetworkModel.real_network())

        # TODO CREATE A WAY TO CUSTOMIZE SAVE LOCATION NOT JUST BASED ON NETWORK ID
        unmatched_nodes_inset, mds = network_cn.control.snap_find_mds_minimum_driver_node_set(
            draw_label=draw_label_prefix
        )
        unmatched_nodes = [unmatched_node[1] for unmatched_node in unmatched_nodes_inset]

        return mds

    @staticmethod
    def identify_node_types(
            networkx_digraph, root_folder_work, debug=False, draw_graphs=False, show_plots=False,
            network_id=1, draw_label_prefix=''):
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
        unmatched_nodes_inset, mds = network_cn.control.snap_find_mds_minimum_driver_node_set(
            draw_label=draw_label_prefix
        )
        unmatched_nodes = [unmatched_node[1] for unmatched_node in unmatched_nodes_inset]

        redundant_nodes, intermittent_nodes, critical_nodes, augmenting_path_list, other_output = \
            network_cn.control.snap_find_redundant_intermittent_critical_nodes()

        perc_r = round(float(len(redundant_nodes)) / len(networkx_digraph.nodes()), 3)
        perc_c = round(float(len(critical_nodes)) / len(networkx_digraph.nodes()), 3)
        perc_i = round(float(len(intermittent_nodes)) / len(networkx_digraph.nodes()), 3)

        print ("   -->N-r: **{} N-c: {} time took to process: {} seconds".format(
            perc_r,
            perc_c,
            tools.get_time_difference_seconds(start_time_2)
        ))

        other_output['experiment'] = ex
        other_output['network_cn'] = network_cn
        other_output['augmenting_path_list'] = augmenting_path_list
        other_output['perc_c'] = perc_c
        other_output['perc_i'] = perc_i
        other_output['unmatched_nodes'] = unmatched_nodes

        if draw_graphs is True:
            GeneralTools.draw_bipartite_rep_graph(
                ex=ex, snap_graph=other_output['bipartite_representation_tungraph'], graph_type='undirected',
                file_name="bipartite_red_colored", network_cn=network_cn, augmenting_path_list=augmenting_path_list,
                original_graph_nodes=networkx_digraph.nodes(),
                label_draw="Unmatched: {} Matched: {} Do -/+{} to get outset/inset\nRedundant:{} Intermittent: {} Critical: {}".format(
                    ["{}|{}".format(node, node + len(networkx_digraph) + 1) for node in mds],
                    ["{}|{}".format(node, node + len(networkx_digraph) + 1) for node in
                     set(networkx_digraph.nodes()) - set(unmatched_nodes)],
                    len(networkx_digraph) + 1, redundant_nodes, intermittent_nodes, critical_nodes
                )
            )

        return perc_r, redundant_nodes, intermittent_nodes, critical_nodes, mds, other_output

    @staticmethod
    def draw_bipartite_rep_graph(ex, network_cn, augmenting_path_list, file_name, original_graph_nodes,
                                 graph_type='undirected', on_before_draw=None, networkx_graph=None, snap_graph=None,
                                 label_draw=''):

        if snap_graph is not None:
            networkx_graph = ex.snap_to_networkx_cnetwork(
                snap_g=snap_graph, name=network_cn.name,
                network_id=network_cn.network_id, model=network_cn.model, graph_type=graph_type).graph

        # You will node to consider the other 3 types of graphs as well
        if graph_type == 'multigraph' and type(networkx_graph) != nx.MultiGraph:
            networkx_multi_graph = nx.MultiGraph()
            networkx_multi_graph.add_nodes_from(networkx_graph.nodes())
            networkx_multi_graph.add_edges_from(networkx_graph.edges(), style="bold")
            del networkx_graph
            networkx_graph = networkx_multi_graph
            # for edge in networkx_graph.edges():
            #     networkx_multi_graph.add_edge(edge[0], edge[1])

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

        positions = {}
        n = len(original_graph_nodes)
        for node in original_graph_nodes[:]:
            original_graph_nodes.append(node + n + 1)

        for node in original_graph_nodes:
            if networkx_graph.has_node(node) is False:
                networkx_graph.add_node(node)

            if node > n:
                positions[node] = "{},0!".format((node - (n + 1)) * 2)
            else:
                positions[node] = "{},4!".format(node * 2)
                pass

        return tools.networkx_draw(
            G=networkx_graph,
            # path="%s/%s/%s.jpg" % (constants.path_draw_graphs, network_cn.network_id, file_name)
            path="%s/%s/%s.jpg" % (constants.path_draw_graphs, ex.root_folder_work, file_name),
            label="{}\nAugmenting Path: {}".format(label_draw, str(augmenting_path_list).replace('], [', ']\n[')),
            positions=positions
        )

    @staticmethod
    def copy_graphs_with_respecting_to_redundant_percentage(from_path, to_path):
        if not os.path.exists(from_path):
            return "frompath does not exists"

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

    @staticmethod
    def filter_and_copy_generated_graphs():
        for i in range(5, 31):
            number = 'n_' + str(i)
            GeneralTools.copy_graphs_with_respecting_to_redundant_percentage(
                from_path="D:\\Temp\\random_graph\\" + number,
                to_path="D:\\Temp\\low_r\\" + number
            )
