import os
import networkx as nx
from collections import Counter
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.core.algorithms as algos
import complex_networks.control as C
import snap
import random
# from .control import Control
import operator
import constants
# os.path.dirname(os.path.abspath(__file__)) # --> 'D:\\SoftwareProject\\network_analysis_service\\complex_networks'


# _work_path = os.path.abspath(_work_path) + os.sep


class Network:

    """

    """

    def __init__(
            self,
            experiment,
            model,
            network_id,
            directed=True,
            name=None,
            nx_graph=None):
        """

        :param experiment:
        :param network_id:
        :param name:
        :param directed:
        :param model:
        :param nx_graph:
        """

        if nx_graph is None:
            self.graph = nx.DiGraph()

            if directed is True:
                self.graph = nx.DiGraph()
            else:
                self.graph = nx.Graph()
        else:
            self.graph = nx_graph

        self.experiment = experiment
        self.control = C.Control(self)

        self.network_id = network_id
        self.name = name
        self.directed = directed
        self.model = model

        self.p_value = None
        self.number_of_nodes = None

    @staticmethod
    def find_longest_path(matrix, n, node):
        node_len = [0 for i in range(n)]
        node_to_visit = [[node, 1]]
        max_node_len = -1
        last_node_in_longest_path = 0

        while node_to_visit:
            curNode = node_to_visit.pop()
            nodeIdx = curNode[0]
            pathLen = curNode[1]

            if (node_len[nodeIdx] != 0):
                continue

            node_len[nodeIdx] = pathLen

            if pathLen > max_node_len:
                max_node_len = pathLen
                last_node_in_longest_path = nodeIdx

            for i in range(n):

                if matrix[i][nodeIdx] == 1 and node_len[i] == 0:
                    node_to_visit.append([i, pathLen + 1])

        path = [last_node_in_longest_path]
        curNode = last_node_in_longest_path
        curLen = max_node_len

        while curNode != node:
            for i in range(n):
                # print matrix[curNode][i] ," ", node_len[i] == curLen - 1

                if matrix[curNode][i] == 1 and node_len[i] == curLen - 1:
                    curNode = i
                    curLen -= 1
                    path.insert(0, curNode)

                    break
        return path

    @staticmethod
    def find_duplicate_edges(graph_edges):
        out_edge = dict([(e[0], []) for e in graph_edges])
        for e in graph_edges:
            out_edge[e[0]].append(e[1])

        not_unique = []
        for out_node, in_nodes in out_edge.items():
            is_unique = len(out_edge[out_node]) - len(set(out_edge[out_node]))
            if is_unique > 0:
                not_unique.append((out_node, is_unique))

        return not_unique

    def average_degree_network(self, type='total'):
        # todo this is wrong (wrong implimentation of calculateing average degree from degree distrifix it later just use L / N (for directed networks)
        """

        :param type:in|out|total
        :return:
        """
        degree = None
        if type == 'total':
            degree = self.graph.degree()
        if type == 'in':
            degree = self.graph.in_degree()
        if type == 'out':
            degree = self.graph.out_degree()

        degree_sequence = [d for n, d in degree.items()]

        c = collections.Counter(degree_sequence)

        n = self.graph.number_of_nodes()

        return round(sum([d * (count / n) for d, count in c.items()]), 4)

        pass

    def adj_matrix(self):
        return nx.adj_matrix(self.graph).todense().tolist()

    def driver_nodes_maximal_matching(self, draw=False):

        control_nodes = C.Control.find_control_nodes(self.graph, draw=draw)

        # print("control_nodes:" + str(control_nodes))
        # print("len(control_nodes):" + str(len(control_nodes)))

        return control_nodes

    def degree_distribution(self):
        Network.degree_distribution_calculate(self.graph, draw=True)
        pass

    def draw_degree_distribution_quartile(self):
        init = [(1, -1, -1), (2, -1, -1), (3, -1, -1), (4, -1, -1)]

        def normalize_counter_values(c):
            # result = map(sub, l, [1, 1, 1, 1])
            result = [c[1] - 1, c[2] - 1, c[3] - 1, c[4] - 1]
            return result

        # test = ComplexNetwork.tiles_degree(self.graph.in_degree(), return_dict=True)

        # if is True returns[(quartile, node, degree)] otherwise: {node: (quartile, degree)}
        in_degree_quartile = init + Network.tiles_degree(self.graph.in_degree())
        out_degree_quartile = init + Network.tiles_degree(self.graph.out_degree())

        counter_for_in_degree = Counter(elem[0] for elem in in_degree_quartile)
        counter_for_out_degree = Counter(elem[0] for elem in out_degree_quartile)

        control_nodes = self.driver_nodes_maximal_matching(draw=True)

        in_degree_control_quartile = [i for i in in_degree_quartile if i[1] in control_nodes] + init
        out_degree_control_quartile = [i for i in out_degree_quartile if i[1] in control_nodes] + init

        raw_data = {'first_name': ['1st', '2nd', '3rd', '4th'],
                    'in_degree': normalize_counter_values(counter_for_in_degree),
                    'ctr_num': normalize_counter_values(Counter(elem[0] for elem in in_degree_control_quartile)),
                    'out_degree': normalize_counter_values(counter_for_out_degree),
                    'out_ctr_num': normalize_counter_values(Counter(elem[0] for elem in out_degree_control_quartile))}

        df = pd.DataFrame(raw_data, columns=['first_name', 'in_degree', 'ctr_num', 'out_degree', 'out_ctr_num'])

        # Setting the positions and width for the bars
        pos = list(range(len(df['in_degree'])))

        width = 0.20

        # Plotting the bars
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a bar with in_degree data,
        # in position pos,

        plt.bar(pos,
                # using df['in_degree'] data,
                df['in_degree'],
                # of width
                width,
                # with alpha 0.5
                alpha=0.5,
                # with color
                color='#EE3224',
                # with label the first value in first_name
                label=df['first_name'][0])

        # Create a bar with ctr_num data,
        # in position pos + some width buffer,
        plt.bar([p + width for p in pos],
                # using df['ctr_num'] data,
                df['ctr_num'],
                # of width
                width,
                # with alpha 0.5
                alpha=0.5,
                # with color
                color='#FFC924',
                # with label the second value in first_name
                label=df['first_name'][1])

        # Create a bar with out_degree data,
        # in position pos + some width buffer,
        plt.bar([p + width * 2 for p in pos],
                # using df['out_degree'] data,
                df['out_degree'],
                # of width
                width,
                # with alpha 0.5
                alpha=0.5,
                # with color
                color='#E157C9',
                # with label the third value in first_name
                label=df['first_name'][2])

        # Create a bar with out_degree data,
        # in position pos + some width buffer,
        plt.bar([p + width * 3 for p in pos],
                # using df['out_degree'] data,
                df['out_ctr_num'],
                # of width
                width,
                # with alpha 0.5
                alpha=0.5,
                # with color
                color='#32AD5A',
                # with label the third value in first_name
                label=df['first_name'][2])

        # Set the y axis label
        ax.set_ylabel('Number of nodes')

        # Set the chart's title

        ax.set_title('Network model: %s  %s' % (self.model, self.name))

        # Set the position of the x ticks
        ax.set_xticks([(p + 4.0 * width) - 1.0 for p in pos])

        # Set the labels for the x ticks
        ax.set_xticklabels(df['first_name'])

        # Setting the x-axis and y-axis limits
        plt.xlim(min(pos) - width, max(pos) + width * 4)
        plt.ylim([0, max(df['in_degree'] + df['ctr_num'] + df['out_degree'])])

        # max_deg = max(df['in_degree'].max(), df['ctr_num'].max(), df['out_degree'].max(), df['out_ctr_num'].max())
        # plt.yticks(np.arange(
        #     0,
        #     max_deg + 10,
        #     # round(max_deg / round((max_deg / 60))))
        #     round(max_deg / 10)
        # ))

        # Adding the legend and showing the plot
        plt.legend(['In-Degree', 'In-Deg ctrl #', 'Out-Degree', 'Out-Deg ctrl #'], loc='upper right')
        plt.grid()
        plt.show()

    def quartile_degree2(self, degree_dict):
        """

        :param degree_dict:
        :return: [(quartile, node, degree)]
        """
        # degree_sequence = sorted([d for n, d in graph.in_degree().items()], reverse=True)  # degree sequence
        degree_sequence = [d for n, d in degree_dict.items()]

        quartiles = pd.qcut(degree_sequence, 4, labels=["1st", "2nd", "3rd", "4th"])
        # quartiles.categories
        # quartiles.value_counts()
        # quartiles.describe()
        # quartiles.get_values()

        quartile_node_degree = zip(quartiles.get_values(), degree_dict.keys(),
                                   degree_dict.values())  # (quartile, node, degree)

        return quartile_node_degree

    @staticmethod
    def tiles_degree(degree_dict, number_of_cuts=4, return_dict=False):
        """

        :param  degree_dict: {node: degree}
        :param number_of_cuts: if set to 4 it will cut the input to quartiles
        :param return_dict: determine the return format
        :return: if return_dict is True returns [(tile, node, degree)] otherwise: {node: (tile, degree)}
        """
        # degree_sequence = sorted([d for n, d in graph.in_degree().items()], reverse=True)  # degree sequence
        degree_sequence = [d for n, d in degree_dict.items()]

        labels = [i for i in range(1, number_of_cuts + 1)]

        # solution from: http://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges

        bins = algos.quantile(degree_sequence, np.linspace(0, 1, number_of_cuts + 1))

        # todo investigate this approach more, but this is not making the ranges that I need (kollan fek nemikonam dorost bashe)
        # I dont need to have np.unique(degree_sequence) because we want to break data into multiple tilels
        # tiles = pd.tools.tile._bins_to_cuts(degree_sequence, bins,
        #                                     include_lowest=True,
        #                                     labels=labels)
        # tiles_values = tiles.get_values().tolist()

        tile_ranges = []
        for i in range(0, number_of_cuts):
            tile_ranges.append([bins[i], bins[i + 1]])

        tile_ranges[0][0] -= 1
        tile_ranges[len(tile_ranges) - 1][1] += 1

        tiles_values = []
        i = 0
        for degree in degree_sequence:
            for j in range(0, number_of_cuts):
                if degree >= tile_ranges[j][0] and degree < tile_ranges[j][1]:
                    tiles_values.append(j + 1)  # number of the tile
                    break
            i += 1

        if return_dict is True:
            # need to use import itertools as it | below implimentations is wring because dict()
            # wont do group by
            # return dict(zip(degree_dict.keys(), zip(tiles_values, degree_dict.values())))
            return None
        else:
            tiles_node_degree = zip(tiles_values, degree_dict.keys(),
                                    degree_dict.values())  # (tile, node, degree)

            return list(tiles_node_degree)

    @staticmethod
    def degree_distribution_calculate(graph, draw=False):
        title = ""

        width = 0.3

        fig, ax = plt.subplots()

        degree_sequence = sorted([d for n, d in graph.in_degree().items()], reverse=True)  # degree sequence
        degree_count = collections.Counter(degree_sequence)
        in_deg, cnt = zip(*degree_count.items())
        in_deg = np.array(in_deg)
        rects1 = ax.bar(in_deg, cnt, width=width, color='b')

        degree_sequence = sorted([d for n, d in graph.out_degree().items()], reverse=True)  # degree sequence
        degree_count = collections.Counter(degree_sequence)
        out_deg, cnt = zip(*degree_count.items())
        out_deg = np.array(out_deg)
        rects2 = ax.bar(out_deg + width, cnt, width=width, color='r')

        degree_sequence = sorted([d for n, d in graph.degree().items()], reverse=True)  # degree sequence
        degree_count = collections.Counter(degree_sequence)
        total_deg, cnt = zip(*degree_count.items())
        total_deg = np.array(total_deg)
        rects3 = ax.bar(total_deg + (width * 2), cnt, width=width, color='g')

        ax.set_ylabel('Count')
        ax.set_title("%s Histogram" % title)
        ax.set_xticks(np.arange(total_deg.max() + 3))  # + width / 2
        # ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))

        ax.legend((rects1, rects2, rects3), ('in-degree', 'out-degree', 'total-degree'))

        # plt.show()

        return in_deg, out_deg, total_deg

    @staticmethod
    def find_longest_paths_starting_from_node(input_graph, source_node):
        if source_node not in input_graph.nodes():
            return "node-not-in-graph"

        paths = []

        for target_node in input_graph.nodes():
            if target_node != source_node:
                paths_to_target = nx.all_simple_paths(input_graph, source=source_node, target=target_node)

                paths += list(paths_to_target)

                continue

        path_lengths = {}

        for v in paths:
            l = len(v)
            if l not in path_lengths:
                path_lengths[l] = []

            path_lengths[l].append(v)

        if len(path_lengths) > 0:
            return path_lengths[max(path_lengths)]

        return []

    @staticmethod
    def find_longest_shortest_paths(input_graph, source_node):

        paths = nx.shortest_path(input_graph, source=source_node)
        del paths[source_node]

        path_lengths = {}

        for v in paths.values():
            l = len(v)
            if l not in path_lengths:
                path_lengths[l] = []

            path_lengths[l].append(v)

        if len(path_lengths) > 0:
            return path_lengths[max(path_lengths)]

        return []

    def generate(
            self,
            n,
            p=0.1,
            remove_loops=True):

        if self.model == constants.NetworkModel.erdos_renyi():
            self.graph = nx.fast_gnp_random_graph(n=n, p=p, directed=self.directed)
        elif self.model == constants.NetworkModel.scale_free():
            self.graph = nx.scale_free_graph(n=n)
        else:
            return "failed"

        if self.directed is True:
            self_loop_edges = self.graph.selfloop_edges()
            # print("[NetworkModel.generate] # self-loop edges: " + str(len(self_loop_edges)))
            self.graph.remove_edges_from(self_loop_edges)

        self.p_value = p
        self.number_of_nodes = n

        self.name = "%s|n=%s p=%s" % (self.name, str(n), str(p))

        return "successful"

    @staticmethod
    def create_from_gml(path, experiment=None, db_id=None, name=""):

        g = nx.read_gml(path, label='id')

        complex_network = Network(
            experiment=experiment,
            network_id=db_id,
            name=name,
            directed=True,
            model=constants.NetworkModel.real_network(),  # make both frameworks use the same model
            p_value=0
        )

        complex_network.graph = g

        # g.node[0] --> {'label': '1'}
        # nx.get_node_attributes(g, 'label')
        # g.nodes(data=True)
        # g[111][86]     --> {'value': 7}
        # g.adj[111][86] --> {'value': 7}

        return complex_network

    @staticmethod
    def create_from_text_file(
            path,
            experiment=None,
            db_id=None,
            name="",
            from_to_separator='\t'
    ):
        g = nx.DiGraph()

        data_source_size_bytes = os.path.getsize(path)
        read_bytes = 0.0
        # todo [DUPLICATED EDGES] you can have it as a set set() instead of [] so it will prevent duplicate edges
        batch_edges = []
        number_of_edges_added = 0

        with open(path) as f:
            for line in f:
                if line.startswith('#'):
                    continue

                ix = line.index(from_to_separator)

                # in windows its \r\n for enter, python only reads \r

                batch_edges.append((
                    int(line[0:ix]),
                    int(line[ix:])
                ))

                read_bytes += len(line) + 1.0

                if len(batch_edges) % 1000 == 0:
                    g.add_edges_from(batch_edges)
                    number_of_edges_added += len(batch_edges)
                    batch_edges.clear()
                    print('[IMPORT FILE] %f%% of file added. number of edges added: %d' % (
                        round(read_bytes / data_source_size_bytes, 2) * 100,
                        number_of_edges_added
                    ))

        g.add_edges_from(batch_edges)
        number_of_edges_added += len(batch_edges)
        batch_edges.clear()
        print('[IMPORT FILE] LAST BATCH: %f%% of file added. number of edges added: %d' % (
            round(read_bytes / data_source_size_bytes, 2) * 100,
            number_of_edges_added
        ))

        complex_network = Network(
            nx_graph=g,
            experiment=experiment,
            network_id=db_id,
            name=name,
            directed=True,
            model=constants.NetworkModel.real_network()  # make both frameworks use the same model
        )

        return complex_network
