from complex_networks.network import Network
import constants
import os
import os.path
import snap


class Experiment:
    def __init__(self, db_experiment_id=None):
        """

        :param db_experiment_id: not used yet maybe in future
        """
        self.experiment_id = None
        self.description = None
        self.create_date = None
        self.networks = {}

        self.db_experiment_id = db_experiment_id

        self.erdos_renyi = {}

    def snap_load_network(self, graph_path, name, network_id, directed=True,
                          model=constants.NetworkModel.real_network()):

        q = Network(
            experiment=self,
            model=model,
            name=name,
            network_id=network_id,
            directed=directed
        )

        path_t = graph_path
        if os.path.isfile(graph_path) is False:
            path_t = os.path.abspath(constants.path_work + graph_path)

        if directed is True:
            snap_directed_graph = snap.LoadEdgeList(
                snap.PNGraph,
                path_t, 0, 1)
        else:
            snap_directed_graph = snap.LoadEdgeList(
                snap.PUNGraph,  # PNEANet -> load directed network  |  PUNGraph -> load directed graph
                path_t, 0, 1)

        print ("[graph loaded] directed: %s Nodes: %d Edges: %d (directed: PNGraph | undirected: PUNGraph)"
               % (str(directed), snap_directed_graph.GetNodes(), snap_directed_graph.GetEdges()))

        q.graph = snap_directed_graph

        # for binary
        # FIn = snap.TFIn(graph_path)
        # Graph = snap.TNGraph.Load(FIn)
        # print(FIn.Len())
        # print snap_directed_graph

        return q

    def generate_network(self, model, n, p, name='', directed=True, remove_loops=True):
        if constants.NetworkModel.parse(model) is None:
            return "not defined network model"

        c_network = Network(
            experiment=self,
            network_id=-1,
            name=name,
            directed=directed,
            model=model,
        )

        if c_network.generate(n=n, p=p, remove_loops=remove_loops) == "successful":
            self.networks[c_network.network_id] = c_network

        return c_network


if __name__ == "__main__":
    path = 'D:\\SoftwareProject\\complex_networks_tools\\data\\Neural Network\\celegansneural.gml'

    net = Network.create_from_gml(path)
    # exp = Experiment()
    # n = 250
    # # p = 0.01
    # network = exp.generate_network(model=constants.NetworkModel.erdos_renyi(), n=n, p=0.01)
    #
    # print(exp.experiment_heuristic_RENAME(network, distribution_type="in"))

    # network, save_status = exp.generate_network(model=constants.NetworkModel.scale_free(), n=1000, p=0.01, save=False)

    # network.degree_distribution()
    # network.quartile()
    # control_nodes = network.driver_nodes_maximal_matching(draw=True)
    # network.draw_degree_distribution_quartile()

    # longes_paths = ComplexNetwork.find_longest_paths_starting_from_node(network.graph,
    #                                                                     random.choice(network.graph.nodes()))

    # longes_shortest_paths = ComplexNetwork.find_longest_shortest_paths(network.graph,
    #                                                                    random.choice(network.graph.nodes()))

    # print(longes_shortest_paths)

    # if is True returns[(quartile, node, degree)] otherwise: {node: (quartile, degree)}


    pass
