import subprocess as s
from subprocess import Popen
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt
import random
# todo seems like lots of work... avoid circular import, fix this
import complex_networks as CN
# CN.complex_network.ComplexNetwork
import constants
from collections import Counter
import itertools as it
import datetime
import snap


class Control:
    def __init__(self, network):

        self.network = network
        pass

    @staticmethod
    def di_graph_bipartite_representation(di_graph):
        b_graph = nx.Graph()
        out_set = []
        in_set = []

        print("[Bipartite Representation] START %s" % str(datetime.datetime.now()))

        for node in di_graph.nodes():
            out_set.append("%i-" % node)

            in_set.append("%i+" % node)

        b_graph.add_nodes_from(out_set, bipartite=0)
        b_graph.add_nodes_from(in_set, bipartite=1)

        b_graph_edges = []
        for edges in di_graph.edges():
            # a = "%i+" % edges[0]
            # b = "%i-" % edges[1]

            # mmeory error here
            b_graph_edges.append(("%i-" % edges[0],
                                  "%i+" % edges[1]))

        b_graph.add_edges_from(b_graph_edges)

        print("[Bipartite Representation] DONE %s" % str(datetime.datetime.now()))

        # not returning correct sets, why??
        # A, B = bipartite.sets(bp) --> not returning correct sets, why?! so I just handle them manually
        # out_set = set(n for n, d in b_graph.nodes(data=True) if d['bipartite'] == 0)
        # in_set = set(b_graph) - out_set

        return out_set, in_set, b_graph

    @staticmethod
    def find_control_nodes(input_di_graph, remove_loops=True, draw=True, debug=False):

        def normalize_node_name(name):
            return int(name.replace('-', '').replace('+', ''))

        if debug is True:
            print(input_di_graph.edges())

        # this is baad, the loops are already removed
        # if remove_loops is True:
        #     duplicates = set([e for e in input_di_graph.edges() if (e[1], e[0]) in input_di_graph.edges()])
        #     duplicates = duplicates - set(tuple(sorted(l)) for l in duplicates)
        #     input_di_graph.remove_edges_from(duplicates)

        if debug is True:
            print("Edges: " + str(input_di_graph.edges()))

        out_set, in_set, bipartite_graph = \
            Control.di_graph_bipartite_representation(input_di_graph)

        matching_algorithm_result = nx.bipartite.hopcroft_karp_matching(bipartite_graph)

        un_matched_in_set_nodes = []
        for in_set_node in in_set:
            if in_set_node not in matching_algorithm_result:
                un_matched_in_set_nodes.append(int(in_set_node[:-1]))

        driver_node_list = un_matched_in_set_nodes

        # matchings_of_the_in_set = []
        # for matching_tuple in  matching_algorithm_result:
        #     if matching_tuple[1][:-1] == '+':
        #         matchings_of_the_in_set.append(matching_tuple[1])
        #         pass

        # digraph_matching_edges = [(normalize_node_name(k), normalize_node_name(v)) for k, v in
        #                           matching_algorithm_result.items() if
        #                           k.endswith("+")]

        # A_vertices_in_matched_edge = set(np.intersect1d(list(matching_algorithm_result.keys()), list(out_set)).tolist())
        # B_vertices_in_matched_edge = set([matching_algorithm_result[m] for m in A_vertices_in_matched_edge])
        #
        # # Im not using node names as string because the plot did not work fine
        # matched_vertices = set([int(n.replace('-', '')) for n in B_vertices_in_matched_edge])
        # unmatched_vertices = set(input_di_graph.nodes()) - matched_vertices

        # control_nodes = unmatched_vertices

        # # # # # # #

        if draw is True:
            if len(bipartite_graph.nodes()) < 21:  # do not draw the Bipartite Equivalent
                plt.figure("Bipartite Equivalent")

                pos = dict()
                pos.update((n, (1, i)) for i, n in enumerate(sorted(out_set, reverse=True)))  # put nodes from X at x=1
                pos.update((n, (2, i)) for i, n in enumerate(sorted(in_set, reverse=True)))  # put nodes from Y at x=2
                edge_colors = []
                for e in bipartite_graph.edges():
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
                for n in bipartite_graph.nodes():
                    sizes.append(1000)

                nx.draw(bipartite_graph, pos=pos, with_labels=True, edge_color=edge_colors, node_size=sizes)

            # # # # # # #
            plt1 = plt.figure("directed graph")
            colors = []
            sizes = []

            for n in input_di_graph.nodes():
                sizes.append(1000)
                # average_degree_connectivity(G, source='in+out', target='in+out', nodes=None, weight=None)
                if n in driver_node_list:
                    colors.append('white')
                else:
                    colors.append('red')

            nx.draw(input_di_graph, with_labels=True, node_color=colors, node_size=sizes)
            # # # # # # #

            # # # # # # #
            if len(bipartite_graph.nodes()) < 51:  # do not draw the Independent Paths
                plt.figure("independent paths")
                graph2 = nx.DiGraph()
                graph2.add_nodes_from(input_di_graph.nodes())
                graph2.add_edges_from(digraph_matching_edges)

                colors = []
                sizes = []

                for n in input_di_graph.nodes():
                    sizes.append(1000)
                    # average_degree_connectivity(G, source='in+out', target='in+out', nodes=None, weight=None)
                    if n in driver_node_list:
                        colors.append('white')
                    else:
                        colors.append('red')

                nx.draw(graph2, with_labels=True, node_color=colors, node_size=sizes)
                # # # # # # #

                # plt.show()

        if debug is True:
            print("A: %s" % list(out_set))
            print("B: %s" % list(in_set))
            # print("Matching Algorithm: %s" % str(matching_algorithm_result))
            print("Digraph matching edges: %s" % str(digraph_matching_edges))
            print("A_matched: %s" % A_vertices_in_matched_edge)
            print("B_matched: %s" % B_vertices_in_matched_edge)

            print("Len: %i Control nodes: %s" % (len(driver_node_list), sorted(driver_node_list)))

        return driver_node_list

    def snap_di_graph_bipartite_representation(self):

        # 999,999,999 is max node number I could get here, I dont know if its same using c++ but seems not
        n = self.network.graph.GetNodes()

        # b_graph = snap.TNEANet.New(
        b_graph = snap.TNEANet.New(  # TUNGraph is undirected
            n * 2,
            self.network.graph.GetEdges(),
        )
        # I guess no need
        out_set = snap.TIntV()
        in_set = snap.TIntV()

        print("[Bipartite Representation] START %s" % str(datetime.datetime.now()))

        source_id = b_graph.AddNode(constants.max_flow_source_id)
        sink_id = b_graph.AddNode(constants.max_flow_sink_id)

        for node in self.network.graph.Nodes():
            node_id = node.GetId()

            b_graph.AddNode(node_id)  # out_set (edge.GetSrcNId())
            out_set.Add(node_id)

            in_set_node_id = n + node_id

            b_graph.AddNode(in_set_node_id)  # in_set (edge.GetDstNId())
            in_set.Add(in_set_node_id)

        for edge in self.network.graph.Edges():
            # edges are from out_set to in_set

            bipartite_edge_source_node = edge.GetSrcNId()  # belongs out_set
            bipartite_edge_destination_node = n + edge.GetDstNId()  # belongs in_set

            b_graph.AddEdge(source_id, bipartite_edge_source_node)
            b_graph.AddEdge(bipartite_edge_source_node, bipartite_edge_destination_node)
            b_graph.AddEdge(bipartite_edge_destination_node, sink_id)

        # b_graph.AddIntAttrDatE(ed_id, 1, "capacity") # -> snap.SaveEdgeList not gonna save the edge attribute (capacity)

        return out_set, in_set, b_graph

    def snap_find_control_nodes(self):
        out_set, in_set, b_graph = self.snap_di_graph_bipartite_representation()

        path_bipartite_representation = self.get_path_bipartite_representation()

        snap.SaveEdgeList(b_graph, path_bipartite_representation, "Save as tab-separated list of edges")

        path_matching_result = self.get_path_matching()

        path_matching = "%s -i:%s -o:%s -src:%i -snk:%i" % \
                        (constants.path_bipartite_matching_snap,
                         path_bipartite_representation,
                         path_matching_result,
                         constants.max_flow_source_id,
                         constants.max_flow_sink_id
                         )

        print (path_matching)

        ps = Popen(path_matching, stdin=s.PIPE, stdout=s.PIPE)
        print ('pOpen done..')

        (stdout, stderr) = ps.communicate()

        print (stdout)
        print (stderr)

        bipartite_network = self.network.experiment.snap_load_network(
            graph_path=path_matching_result,
            model=constants.NetworkModel.bipartite_matching(),
            name='BIPARTITE MATCHING - ' + self.network.name,
            network_id=0,
            directed=False
        )

        n = self.network.graph.GetNodes()
        control_nodes = [node - n for node in in_set if bipartite_network.graph.IsNode(node) is False]

        file_cn = open(self.get_path_control_nodes(), 'w')
        file_cn.write("\n".join(str(x) for x in control_nodes))
        file_cn.close()

        return {
            "path_matching": path_matching,
            "control_nodes": control_nodes,
            "path_control_nodes": self.get_path_control_nodes()
        }

    def get_path_control_nodes(self):
        return os.path.abspath(
            "%s%i_control_nodes.txt" % (constants.path_bipartite_representations, self.network.network_id)
        )

    def get_path_matching(self):
        return os.path.abspath(
            "%s%i_matching.txt" % (constants.path_bipartite_representations, self.network.network_id)
        )

    def get_path_bipartite_representation(self):

        return os.path.abspath(
            "%s%i_bipartite_representation.txt" % (constants.path_bipartite_representations, self.network.network_id)
        )

    @staticmethod
    def percentage_control_forough(matrix, n, driver_nodes):

        paths = []

        # n * 0.2 --> i think it means first quartile
        for i in range(len(driver_nodes)):
            if paths:
                found = False
                for path in paths:
                    for node in path:
                        if node == driver_nodes[i]:
                            found = True
                            break
                        if found:
                            break
                if found:
                    continue

            paths.append(CN.complex_network.ComplexNetwork.find_longest_path(matrix, n, driver_nodes[i]))

            for node in paths[-1]:
                for i in range(n):
                    matrix[node][i] = 0
                    matrix[i][node] = 0
            # '''
            prev = -1

            for node in paths[-1]:

                if prev == -1:
                    prev = node
                else:
                    # the previous loop removed all edges coming from or going to the each node in the paths[-1]
                    # I believe this condition is never True
                    if matrix[node][prev] != 0:
                        print("hello! check this point")

                    matrix[node][prev] = 0
                    prev = node

        # print("paths: " + str(paths))
        # print("len(paths): " + str(len(paths)))

        nodes_not_in_path = [i for i in range(n)]

        for path in paths:
            for node in path:
                if nodes_not_in_path.count(node) > 0:
                    nodes_not_in_path.remove(node)

        nodes_in_path = []
        for path in paths:
            for node in path:
                if nodes_in_path.count(node) == 0:
                    nodes_in_path.append(node)

        l = (len(nodes_in_path) + 0.0) / n
        # print("percentage of nodes in path = ", l)

        # print("nodes not in path: ")
        # print(nodes_not_in_path)

        return l, nodes_not_in_path

    def get_percentage_control(
            self,
            control_nodes,
            longest_shortest_path=False,
            scale=None):
        if len(self.network.graph.nodes()) == 0:
            return 'graph is empty'

        # todo save the result in database defiantly

        g = self.network.graph.copy()

        for control_node in control_nodes:
            if control_node not in g.nodes():
                continue

            if longest_shortest_path:
                paths = CN.complex_network.CN.find_longest_shortest_paths(
                    g,
                    source_node=control_node)
            else:
                paths = CN.complex_network.CN.find_longest_paths_starting_from_node(
                    g,
                    source_node=control_node)

            if len(paths) > 0:
                path = random.choice(paths)

                g.remove_nodes_from(path)
            else:
                g.remove_node(control_node)

        result = 1 - (len(g.nodes()) / len(self.network.graph.nodes()))

        if scale is not None:
            if scale == 0:
                return -1
            return result / scale

        return result

    @staticmethod
    def degree_tiles_distributions_generate(
            graph,
            number_of_tiles,
            control_nodes,
            include_nodes_tile_list=False,
            percentage=True,
            random_sampling_size=0):

        def del_non_control_nodes_from_degree_dict(degree_dict, ctrl_nodes):
            for k in list(degree_dict.keys()):
                if k not in ctrl_nodes:
                    del degree_dict[k]

            return degree_dict

        def normalize_val(x_y_input, num_of_tiles=4, divide_by=0):

            values = []
            for i in range(1, num_of_tiles + 1):
                values.append({'x': i, 'y': 0})

            for input_val in x_y_input:
                for v in values:
                    if v['x'] == input_val['x']:
                        v['y'] = input_val['y'] / divide_by

            return values

        def find_min_max_degree_tiles(degree_tiles):

            func_result = {}
            # if there is no driver node the reason is all the nodes in the network are matched
            if len(control_nodes) == 0:
                for i in range(1, number_of_tiles + 1):
                    func_result[i] = {'min': 0, 'max': 0}

                return func_result

            tiles_node_degree_lists = list(zip(*degree_tiles))

            quartile_zip_node_degree_list = list(
                zip(tiles_node_degree_lists[0], zip(tiles_node_degree_lists[1], tiles_node_degree_lists[2])))

            tiles_node_degree_dict = {k: tuple(x[1] for x in v) for k, v in
                                      it.groupby(sorted(quartile_zip_node_degree_list), key=lambda x: x[0])}

            for tile, node_degree in tiles_node_degree_dict.items():
                node_degree_sorted_by_degree = sorted(node_degree, key=lambda x: x[1])
                func_result[tile] = {
                    'min': node_degree_sorted_by_degree[0][1],
                    'max': node_degree_sorted_by_degree[len(node_degree_sorted_by_degree) - 1][1]
                }

            return func_result

        def randomly_pick_degree(degree_dict):
            keys = list(degree_dict.keys()).copy()

            randomly_chosen_keys = random.sample(keys, random_sampling_size)
            for key in keys:
                if key not in randomly_chosen_keys:
                    del degree_dict[key]

        number_of_tiles = int(number_of_tiles)

        graph_in_degree = graph.in_degree()
        graph_out_degree = graph.out_degree()
        graph_total_degree = graph.degree()

        if random_sampling_size > 0:
            randomly_pick_degree(graph_in_degree)
            randomly_pick_degree(graph_out_degree)
            randomly_pick_degree(graph_total_degree)

        # if is True returns[(quartile, node, degree)] otherwise: {node: (quartile, degree)}
        in_degree_nodes_tile = CN.complex_network.CN.tiles_degree(
            graph_in_degree, number_of_cuts=number_of_tiles)
        out_degree_nodes_tile = CN.complex_network.CN.tiles_degree(
            graph_out_degree, number_of_cuts=number_of_tiles)
        total_degree_nodes_tile = CN.complex_network.CN.tiles_degree(
            graph_total_degree, number_of_cuts=number_of_tiles)

        counter_for_in_degree_tiles = Counter(elem[0] for elem in in_degree_nodes_tile)
        counter_for_out_degree_tiles = Counter(elem[0] for elem in out_degree_nodes_tile)
        counter_for_total_degree_tiles = Counter(elem[0] for elem in total_degree_nodes_tile)

        # FOR CONTROL NODES

        in_degree_control_nodes = del_non_control_nodes_from_degree_dict(
            graph.in_degree(), control_nodes)
        out_degree_control_nodes = del_non_control_nodes_from_degree_dict(
            graph.out_degree(), control_nodes)
        total_degree_control_nodes = del_non_control_nodes_from_degree_dict(
            graph.degree(), control_nodes)

        in_degree_control_nodes_tile = CN.complex_network.CN.tiles_degree(
            in_degree_control_nodes,
            number_of_cuts=number_of_tiles)
        out_degree_control_nodes_tile = CN.complex_network.CN.tiles_degree(
            out_degree_control_nodes,
            number_of_cuts=number_of_tiles)
        total_degree_control_nodes_tile = CN.complex_network.CN.tiles_degree(
            total_degree_control_nodes,
            number_of_cuts=number_of_tiles)

        counter_for_control_in_degree_tiles = Counter(elem[0] for elem in in_degree_control_nodes_tile)
        counter_for_control_out_degree_tiles = Counter(elem[0] for elem in out_degree_control_nodes_tile)
        counter_for_control_total_degree_tiles = Counter(elem[0] for elem in total_degree_control_nodes_tile)

        number_of_control_nodes = 1
        number_of_nodes = 1
        if percentage is True:
            number_of_control_nodes = len(control_nodes)
            number_of_nodes = len(graph.nodes())

        result = {
            'message': {
                'tiles_max_min_degrees': {
                    'control_in_degree': find_min_max_degree_tiles(in_degree_control_nodes_tile),
                    'control_out_degree': find_min_max_degree_tiles(out_degree_control_nodes_tile),
                    'control_total_degree': find_min_max_degree_tiles(total_degree_control_nodes_tile),
                    'in_degree': find_min_max_degree_tiles(in_degree_nodes_tile),
                    'out_degree': find_min_max_degree_tiles(out_degree_nodes_tile),
                    'total_degree': find_min_max_degree_tiles(total_degree_nodes_tile)
                }
            },
            'in_degree': normalize_val(
                [{'x': k, 'y': v} for k, v in dict(counter_for_in_degree_tiles).items()],
                num_of_tiles=number_of_tiles,
                divide_by=number_of_nodes),
            'control_in_degree': normalize_val(
                [{'x': k, 'y': v} for k, v in dict(counter_for_control_in_degree_tiles).items()],
                num_of_tiles=number_of_tiles,
                divide_by=number_of_control_nodes),

            'out_degree': normalize_val(
                [{'x': k, 'y': v} for k, v in dict(counter_for_out_degree_tiles).items()],
                num_of_tiles=number_of_tiles,
                divide_by=number_of_nodes),
            'control_out_degree': normalize_val(
                [{'x': k, 'y': v} for k, v in dict(counter_for_control_out_degree_tiles).items()],
                num_of_tiles=number_of_tiles,
                divide_by=number_of_control_nodes),

            'total_degree': normalize_val(
                [{'x': k, 'y': v} for k, v in dict(counter_for_total_degree_tiles).items()],
                num_of_tiles=number_of_tiles,
                divide_by=number_of_nodes),
            'control_total_degree': normalize_val(
                [{'x': k, 'y': v} for k, v in dict(counter_for_control_total_degree_tiles).items()],
                num_of_tiles=number_of_tiles,
                divide_by=number_of_control_nodes)
        }

        if include_nodes_tile_list is True:
            result['nodes-tile-distributions'] = {
                'in_degree_control_nodes_tile': in_degree_control_nodes_tile,
                'out_degree_control_nodes_tile': out_degree_control_nodes_tile,
                'total_degree_control_nodes_tile': total_degree_control_nodes_tile,

                'in_degree_nodes_tile': in_degree_nodes_tile,
                'out_degree_nodes_tile': out_degree_nodes_tile,
                'total_degree_nodes_tile': total_degree_nodes_tile
            }

        return result
