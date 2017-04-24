import subprocess as s
import json
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
from datetime import datetime
import snap
import tools as tools


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

    def snap_di_graph_bipartite_representation_reduced_max_flow(self):

        # 999,999,999 is max node number I could get here, I dont know if its same using c++ but seems not
        n = self.network.graph.GetNodes()
        bipartite_max_node_id = self.network.graph.GetMxNId() * 2

        # b_graph = snap.TNEANet.New(
        b_graph = snap.TNEANet.New(  # TUNGraph is undirected
            bipartite_max_node_id,
            self.network.graph.GetEdges(),
        )

        # I guess no need
        out_set = snap.TIntV()
        in_set = snap.TIntV()

        source_node = b_graph.AddNode(bipartite_max_node_id + 1)
        sink_node = b_graph.AddNode(bipartite_max_node_id + 2)

        for node in self.network.graph.Nodes():
            node_id = node.GetId()

            if node_id == 299 or b_graph.IsNode(node_id):
                # print ([n.GetId() for n in b_graph.Nodes()])
                # print "yo"
                assert True, "Why node exists already!!??"

            b_graph.AddNode(node_id)  # out_set (edge.GetSrcNId())
            out_set.Add(node_id)

            in_set_node_id = n + node_id + 1

            b_graph.AddNode(in_set_node_id)  # in_set (edge.GetDstNId())
            in_set.Add(in_set_node_id)

        for edge in self.network.graph.Edges():
            # edges are from out_set to in_set

            bipartite_edge_source_node = edge.GetSrcNId()  # belongs out_set
            bipartite_edge_destination_node = n + edge.GetDstNId() + 1  # belongs in_set | +1 for in case node id is 0

            if b_graph.IsEdge(source_node, bipartite_edge_source_node) is False:
                b_graph.AddEdge(source_node, bipartite_edge_source_node)

            if b_graph.IsEdge(bipartite_edge_source_node, bipartite_edge_destination_node) is False:
                b_graph.AddEdge(bipartite_edge_source_node, bipartite_edge_destination_node)

            if b_graph.IsEdge(bipartite_edge_destination_node, sink_node) is False:
                b_graph.AddEdge(bipartite_edge_destination_node, sink_node)

        # b_graph.AddIntAttrDatE(ed_id, 1, "capacity") # -> snap.SaveEdgeList not gonna save the edge attribute (capacity)

        return out_set, in_set, b_graph, source_node, sink_node

    def snap_find_mds_minimum_driver_node_set(self, draw_label=''):

        # identify critical control nodes
        with open(self.get_path_critical_control_nodes(), 'w') as writefile:
            json.dump([n.GetId() for n in self.network.graph.Nodes() if n.GetInDeg() == 0], writefile)

        out_set, in_set, graph_max_flow_representation, source_node, sink_node = \
            self.snap_di_graph_bipartite_representation_reduced_max_flow()

        path_bipartite_representation_reduced_max_flow = \
            self.get_path_bipartite_representation_reduced_max_flow_graph()

        snap.SaveEdgeList(graph_max_flow_representation, path_bipartite_representation_reduced_max_flow,
                          "Save as tab-separated list of edges")

        if self.network.experiment.draw_graphs:
            # tools.snap_draw(
            #     graph_max_flow_representation,
            #     "%s/%s/b_graph_maxflow" % (constants.path_draw_graphs, self.network.experiment.root_folder_work),
            #     "max flow")
            pass
        # ;;;;;;;;;;;;;;;;;;;;;;;; save bipartite representation ;;;;;;;;;;;;;;;;;;;;
        graph_max_flow_representation.DelNode(source_node)
        graph_max_flow_representation.DelNode(sink_node)

        snap.SaveEdgeList(graph_max_flow_representation, self.get_path_bipartite_representation_graph(),
                          "Save as tab-separated list of edges")
        # FOut = snap.TFOut(self.get_path_bipartite_representation_graph())
        # graph_max_flow_representation.Save(FOut)
        # FOut.Flush()
        # ;;;;;;;;;;;;;;;;;;;;;;;; save bipartite representation ;;;;;;;;;;;;;;;;;;;;

        path_maximum_matching = self.get_path_maximum_matching_graph()

        path_of_matching_calculator_application = "%s -i:%s -o:%s -src:%i -snk:%i" % \
                                                  (
                                                      constants.path_bipartite_matching_app_snap,
                                                      path_bipartite_representation_reduced_max_flow,  # input file
                                                      path_maximum_matching,  # output file
                                                      source_node,
                                                      sink_node
                                                  )

        if self.network.experiment.debug:
            print ("\n[executing started %s] %s" % (str(datetime.now()), path_of_matching_calculator_application))

        ps = Popen(path_of_matching_calculator_application, stdin=s.PIPE, stdout=s.PIPE)

        (stdout, stderr) = ps.communicate()

        # print ("[stdout] --> %s \n [stderr] --> %s" % (stdout, stderr))
        if self.network.experiment.debug:
            print ("[execution done] %s\n" % str(datetime.now()))

        maximum_matching_graph = self.network.experiment.snap_load_network(
            graph_path=path_maximum_matching,
            is_binary_file=False,
            model=constants.NetworkModel.bipartite_matching(),
            name='BIPARTITE MATCHING - ' + self.network.name,
            network_id=self.network.network_id,
            directed=False,
            draw=True
        )

        # if self.network.experiment.debug:
        #     tools.snap_draw(
        #         maximum_matching_graph.graph,
        #         tools.relative_path("/temp/work/draw/maximum_matching_graph.jpg"),
        #         "maximum_matching_graph")

        n = self.network.graph.GetNodes()

        # unmatched nodes are the control nodes
        # Both below will contain tuple (id in bipartite, id in given graph)
        unmatched_nodes_inset = []
        mds = []
        matched_nodes_inset = []

        for node in in_set:
            if maximum_matching_graph.graph.IsNode(node):
                # -1 is important
                matched_nodes_inset.append((node, node - n - 1))
            else:
                node_id = node - n - 1
                unmatched_nodes_inset.append((node, node_id))
                mds.append(node_id)

        with open(self.get_path_control_unmatched_nodes_inset(), 'w') as outfile:
            json.dump(unmatched_nodes_inset, outfile)

        with open(self.get_path_matched_nodes_inset(), 'w') as outfile:
            json.dump(matched_nodes_inset, outfile)

        if self.network.experiment.debug:
            print (
                "\n[# number of control nodes] %f | %s\n     tuples are --> (id in the bipartite representation, id in the given graph)\n" % (
                    round(len(unmatched_nodes_inset) / float(n), 3),
                    str(unmatched_nodes_inset) if len(unmatched_nodes_inset) < 100 else "too big"))

        if self.network.experiment.draw_graphs:
            tools.snap_draw(
                self.network.graph,
                "%s/%s/graph" % (constants.path_draw_graphs, self.network.experiment.root_folder_work),
                draw_label + " Nd: {0} #Matched: {1}\nMDS: {2}\nMatched: {3}".format(
                    len(mds), len(matched_nodes_inset), mds,
                    [matched_node[1] for matched_node in matched_nodes_inset]
                ))

        return unmatched_nodes_inset, mds

    # todo impliment caching
    def snap_load_maximum_matching_cnetwork(self):
        return self.network.experiment.snap_load_network(
            graph_path=self.get_path_maximum_matching_graph(),
            is_binary_file=False,
            model=constants.NetworkModel.bipartite_matching(),
            name='MAXIMUM MATCHING - ' + self.network.name,
            network_id=self.network.network_id,
            directed=False,
            draw=True
        )

    # todo implement caching
    def snap_load_bipartite_representation_cnetwork(self):
        return self.network.experiment.snap_load_network(
            graph_path=self.get_path_bipartite_representation_graph(),
            is_binary_file=False,
            model=constants.NetworkModel.bipartite_matching(),
            name='BIPARTITE REPRESENTATION - ' + self.network.name,
            network_id=self.network.network_id,
            directed=False,
            draw=False
        )

    def snap_is_path_to_unmatched_node_exists(
            self,
            source_node,
            graph,
            maximum_matching_tungraph,
            unmatched_nodes,
            create_bfs_tree=False,
            return_path_to_unmatched_node=False
    ):
        """

        :param source_node:
        :param graph:
        :param maximum_matching_tungraph:
        :param unmatched_nodes:
        :param create_bfs_tree:
        :param return_path_to_unmatched_node:
        :return: True node is intermittent, False if node is redundant
        """
        networkx_graph = None
        if return_path_to_unmatched_node is True:
            networkx_graph = nx.DiGraph()

        # todo instead of bfs tree maintain a vector, no need to waste process power on this
        bfs_tree = snap.PUNGraph.New()
        bfs_tree.AddNode(source_node)

        # special case when no link exists the node is redundant
        if len(list(graph.GetNI(source_node).GetOutEdges())) == 0:
            return False, bfs_tree, None

        # used TIntV (vector) as queue
        queue = snap.TIntV()
        queue.Add(source_node)
        look_for_unmatched_link = True

        while not queue.Empty():

            u = queue[0]
            queue.Del(0)

            for v in graph.GetNI(u).GetOutEdges():

                if bfs_tree.IsNode(v) is False:
                    if look_for_unmatched_link is True:
                        if maximum_matching_tungraph.IsEdge(u, v):
                            # if current link is matched, look for the next link
                            continue

                    if look_for_unmatched_link is False:  # look for matched link
                        if maximum_matching_tungraph.IsEdge(u, v) is False:
                            # id current link is not matched, look for a matched link
                            continue

                    bfs_tree.AddNode(v)
                    if create_bfs_tree is True:
                        bfs_tree.AddEdge(u, v)

                    if return_path_to_unmatched_node is True:
                        networkx_graph.add_edge(u, v)

                    # probably dont need "look_for_unmatched_link is True" but just as an insurance
                    if look_for_unmatched_link is True and v in unmatched_nodes:
                        if return_path_to_unmatched_node is True:
                            return True, bfs_tree, nx.shortest_path(networkx_graph, source_node, v)

                        return True, bfs_tree, None

                    queue.Add(v)

                    look_for_unmatched_link = not look_for_unmatched_link

        # snap.GetShortPath_PUNGraph(bfs_tree, 1, 10)

        # print ([(e.GetSrcNId(), e.GetDstNId()) for e in bfs_tree.Edges()])
        return False, bfs_tree, None

        # return bfs_tree

    def snap_find_redundant_intermittent_critical_nodes(self):
        bipartite_representation_tungraph = self.snap_load_bipartite_representation_cnetwork().graph

        # if self.network.experiment.draw_graphs:
        #     tools.snap_draw(
        #         bipartite_representation_tungraph,
        #         tools.relative_path("/temp/work/draw/bipartite_representation_tungraph.jpg"),
        #         "bipartite_representation_tungraph")
        # tools.snap_draw(bipartite_representation_tungraph, "d:\\wtf.jpg")

        # contains matched links
        maximum_matching_tungraph = self.snap_load_maximum_matching_cnetwork().graph
        # matched_links = [n.GetId() for n in maximum_matching_cnetwork.graph.Nodes()]

        unmatched_nodes, matched_nodes_inset = self.load_inset_matched_and_control_nodes()

        critical_nodes, intermittent_nodes, redundant_nodes = self.load_critical_redundant_intermittent_control_nodes()

        redundant_nodes = []  # snap.TIntV()
        # critical_nodes = snap.TIntV()
        intermittent_nodes = list(set([u[1] for u in unmatched_nodes]) - critical_nodes)  # snap.TIntV()

        augmenting_path_list = []

        for matched_node in matched_nodes_inset:
            matched_node_id = matched_node[0]  # [0] is the inset assigned id, [1] original graph node id
            j = maximum_matching_tungraph.GetNI(matched_node[0]).GetNbrNId(0)

            # i (matched_node_id) neighbors ID ies
            matched_node_neighbors = list(bipartite_representation_tungraph.GetNI(matched_node_id).GetOutEdges())

            bipartite_representation_tungraph.DelNode(matched_node_id)

            is_path_to_unmatched_node_exists, bfs_tree, augmenting_path = self.snap_is_path_to_unmatched_node_exists(
                source_node=j,
                graph=bipartite_representation_tungraph,
                maximum_matching_tungraph=maximum_matching_tungraph,
                unmatched_nodes=[u[0] for u in unmatched_nodes],
                create_bfs_tree=True,
                return_path_to_unmatched_node=True
                # matched_edges=[(e.GetSrcNId(), e.GetDstNId()) for e in maximum_matching_tungraph.Edges()]
            )

            if augmenting_path is not None and len(augmenting_path) > 0:
                augmenting_path_list.append(augmenting_path)

            if is_path_to_unmatched_node_exists:
                intermittent_nodes.append(matched_node[1])
            else:
                redundant_nodes.append(matched_node[1])

            # ;;;;;;;;;;;;;;;;; re-add the deleted node and its edges ;;;;;;;;;;;;
            bipartite_representation_tungraph.AddNode(matched_node_id)

            for matched_node_neighbor in matched_node_neighbors:
                bipartite_representation_tungraph.AddEdge(matched_node_neighbor, matched_node_id)
                # ;;;;;;;;;;;;;;;;; re-add the deleted node and its edges ;;;;;;;;;;;;

        n = bipartite_representation_tungraph.GetNodes() / 2  # its the bipartite representation

        if self.network.experiment.debug:
            print (
                "\n[# number of Redundant nodes] %f | %s" % (
                    round(len(redundant_nodes) / float(n), 3),
                    str(redundant_nodes) if len(redundant_nodes) < 100 else "too big"))
            print (
                "[# number of Intermittent nodes] %f | %s" % (
                    round(len(intermittent_nodes) / float(n), 3),
                    str(intermittent_nodes) if len(intermittent_nodes) < 100 else "too big"))
            print (
                "[# number of Critical nodes] %f | %s" % (
                    round(len(critical_nodes) / float(n), 3),
                    str(critical_nodes) if len(critical_nodes) < 100 else "too big"))

        # print ("redundant_nodes: %s intermittent_nodes: %s critical_nodes: %s" % (
        #     str(list(redundant_nodes)),
        #     str(list(intermittent_nodes)),
        #     str(list(critical_nodes))
        # ))

        # ;;;;;;;;;;;;;;;;;; draw bipartite representation ;;;;;;;;;;;;;;;;;;;

        other_output = {}

        other_output['bipartite_representation_tungraph'] = bipartite_representation_tungraph
        # ;;;;;;;;;;;;;;;;;; end draw graph ;;;;;;;;;;;;;;

        with open(self.get_path_control_augmenting_path_identify_redundant(), 'w') as writefile:
            json.dump(augmenting_path_list, writefile)

        with open(self.get_path_intermittent_control_nodes(), 'w') as writefile:
            json.dump(intermittent_nodes, writefile)

        with open(self.get_path_redundant_control_nodes(), 'w') as writefile:
            json.dump(redundant_nodes, writefile)

        return redundant_nodes, intermittent_nodes, critical_nodes, augmenting_path_list, other_output

    def load_inset_matched_and_control_nodes(self):
        # control nodes are unmatched nodes in the inset

        with open(self.get_path_control_unmatched_nodes_inset(), 'r') as infile:
            control_nodes = json.load(infile)

        with open(self.get_path_matched_nodes_inset(), 'r') as infile:
            matched_nodes_inset = json.load(infile)

        return control_nodes, matched_nodes_inset

    def load_critical_redundant_intermittent_control_nodes(self):
        # control nodes are unmatched nodes in the inset
        critical_nodes = None
        intermittent_nodes = None
        redundant_nodes = None

        if os.path.isfile(self.get_path_critical_control_nodes()):
            try:
                with open(self.get_path_critical_control_nodes(), 'r') as infile:
                    critical_nodes = set(json.load(infile))
            except ValueError:
                critical_nodes = set()

        if os.path.isfile(self.get_path_intermittent_control_nodes()):
            try:
                with open(self.get_path_intermittent_control_nodes(), 'r') as infile:
                    intermittent_nodes = set(json.load(infile))
            except ValueError:
                intermittent_nodes = set()

        if os.path.isfile(self.get_path_redundant_control_nodes()):
            try:
                with open(self.get_path_redundant_control_nodes(), 'r') as infile:
                    redundant_nodes = set(json.load(infile))
            except ValueError:
                redundant_nodes = set()

        return critical_nodes, intermittent_nodes, redundant_nodes

    def get_path_critical_control_nodes(self):
        path = os.path.abspath(
            "%s\\%s\\critical_control_nodes.txt" % (
                constants.path_bipartite_representations, self.network.experiment.root_folder_work,
            )
        )

        tools.create_path_if_not_exists(path)

        return path

    def get_path_redundant_control_nodes(self):
        # return os.path.abspath(
        #     "%s%i_redundant_never_control_nodes.txt" % (
        #         constants.path_bipartite_representations, self.network.network_id)
        # )

        path = os.path.abspath(
            "%s\\%s\\redundant_never_control_nodes.txt" % (
                constants.path_bipartite_representations, self.network.experiment.root_folder_work
            )
        )

        tools.create_path_if_not_exists(path)
        return path

    def get_path_intermittent_control_nodes(self):
        # return os.path.abspath(
        #     "%s%i_intermittent_control_nodes.txt" % (
        #         constants.path_bipartite_representations, self.network.network_id)
        # )

        path = os.path.abspath(
            "%s\\%s\\intermittent_control_nodes.txt" % (
                constants.path_bipartite_representations, self.network.experiment.root_folder_work
            )
        )

        tools.create_path_if_not_exists(path)
        return path

    def get_path_control_augmenting_path_identify_redundant(self):
        # return os.path.abspath(
        #     "%s%i_control_augmenting_path_identify_redundant.txt" % (
        #         constants.path_bipartite_representations, self.network.network_id)
        # )

        path = os.path.abspath(
            "%s\\%s\\control_augmenting_path_identify_redundant.txt" % (
                constants.path_bipartite_representations, self.network.experiment.root_folder_work
            )
        )

        tools.create_path_if_not_exists(path)
        return path

    def get_path_control_unmatched_nodes_inset(self):
        # return os.path.abspath(
        #     "%s%i_control_inset_unmatched_nodes.txt" % (
        #         constants.path_bipartite_representations, self.network.network_id)
        # )

        path = os.path.abspath(
            "%s\\%s\\control_inset_unmatched_nodes.txt" % (
                constants.path_bipartite_representations, self.network.experiment.root_folder_work
            )
        )
        tools.create_path_if_not_exists(path)
        return path

    def get_path_matched_nodes_inset(self):
        # return os.path.abspath(
        #     "%s%i_matched_nodes_inset.txt" % (constants.path_bipartite_representations, self.network.network_id)
        # )
        path = os.path.abspath(
            "%s\\%s\\matched_nodes_inset.txt" % (
                constants.path_bipartite_representations, self.network.experiment.root_folder_work
            )
        )

        tools.create_path_if_not_exists(path)
        return path

    def get_path_maximum_matching_graph(self):
        # return os.path.abspath(
        #     "%s%i_matching.txt" % (constants.path_bipartite_representations, self.network.network_id)
        # )

        path = os.path.abspath(
            "%s\\%s\\matching.txt" % (
                constants.path_bipartite_representations, self.network.experiment.root_folder_work
            )
        )

        tools.create_path_if_not_exists(path)
        return path

    def get_path_bipartite_representation_reduced_max_flow_graph(self):

        # return os.path.abspath(
        #     "%s%i_bipartite_representation_reduced_max_flow.txt" % (
        #         constants.path_bipartite_representations, self.network.network_id)
        # )

        path = os.path.abspath(
            "%s\\%s\\bipartite_representation_reduced_max_flow.txt" % (
                constants.path_bipartite_representations, self.network.experiment.root_folder_work
            )
        )
        tools.create_path_if_not_exists(path)
        return path

    def get_path_bipartite_representation_graph(self):

        # return os.path.abspath(
        #     "%s%i_bipartite_representation.txt" % (
        #         constants.path_bipartite_representations, self.network.network_id)
        # )

        path = os.path.abspath(
            "%s\\%s\\bipartite_representation.txt" % (
                constants.path_bipartite_representations, self.network.experiment.root_folder_work
            )
        )
        tools.create_path_if_not_exists(path)
        return path

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
