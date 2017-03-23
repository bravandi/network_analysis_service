from complex_networks.heuristic import Heuristic
import common.event as event
import random
# from complex_networks.complex_network import NetworkModel, ComplexNetwork
import numpy
import numpy as np


class Heuristic1(Heuristic):
    event_sample_node_post = event.Event("""
    Fires when a nodes is sampled and analyzed.

    :params['node']: The sampled node
    :params['bucket_number']: Bucket/tile number. None ,if the sampled node is not a driver node.
                              Otherwise, the discovered driver node bucket number.
    :params['successful_sampled_nodes_count']: The number of sampled nodes fell into a bucket including this one
    :params['buckets']: Access to exact buckets and nodes inside them. One usage can be calculating percentage
                        of control when a new control node is identified
    :params['number_of_sampled_nodes']: Total number of sampled nodes
    :params['nodes_in_any_bucket']: the nodes in buckets (driver nodes)
    :param['actual_percentage_control']: actual percentage of control using actual driver nodes
    """)
    event_buckets_ranges_created = event.Event("""
    Fires when the buckets are created.

    :param buckets: [{'number': (min, max)}]
    """)

    complex_network = None

    driver_nodes = None

    def __init__(self, degree_distributions, driver_nodes, complex_network=None, handler_buckets_created=None,
                 handler_sample_node_post=None):
        """

        :param degree_distributions: {'in_degree': {node: int}, 'out_degree': {node: int}, 'total_degree': {node: int}}
                                    Do not have to have all three distributions since the heuristic runs only on one
                                    distribution in an execution.
        :param complex_network: for now we dont need it since the degree distributions are fetched from db directly
        :param handler_buckets_created:
        :param handler_sample_node_post:
        """
        Heuristic.__init__(self)

        if handler_buckets_created is not None:
            self.event_buckets_ranges_created += handler_buckets_created
        if handler_sample_node_post is not None:
            self.event_sample_node_post += handler_sample_node_post

        self.degree_distributions = degree_distributions
        self.complex_network = complex_network
        self.driver_nodes = driver_nodes

    def run(self,
            distribution_type,
            distribution_criteria,
            expected_number_of_driver_nodes=0,
            max_samples_number=0):
        # todo instead of complex_network use db_network, I mean probably you do not need to reconstruct the nx.graph
        """

        :param distribution_type: domain in, our, total
        :param distribution_criteria:
        :param expected_number_of_driver_nodes: if equal to 0 then the number of driver nodes will be used
        :return:
        """

        use_longest_shortest_path = True
        result_matrix = []

        # output = ''
        n = len(self.complex_network.graph.nodes())

        number_of_driver_nodes = len(self.driver_nodes)

        if expected_number_of_driver_nodes <= 0:
            expected_number_of_driver_nodes = number_of_driver_nodes

        # todo it is expensive to run this every time for large graphs --> code to use the cn_controlpercentage table
        percent_algorithmic = round(
            self.complex_network.control.get_percentage_control(
                self.driver_nodes,
                longest_shortest_path=True), 2)

        # randomly_pick_number_of_nodes = int((n * 20) / 100)
        # randomly_pick_number_of_nodes = random.randrange(
        #     int((expected_number_of_driver_nodes * 50) / 100),
        #     expected_number_of_driver_nodes + int((expected_number_of_driver_nodes * 25) / 100)
        # )
        # too much work for large graphs
        # percent_random = self.complex_network.control.get_percentage_control(
        #     random.sample(self.complex_network.graph.nodes(), randomly_pick_number_of_nodes),
        #     longest_shortest_path=use_longest_shortest_path,
        #     scale=percent_algorithmic
        # ) * 100
        randomly_pick_number_of_nodes = 'NA'
        percent_random = -1
        result_matrix.append(('just_random_taken_size', randomly_pick_number_of_nodes))
        result_matrix.append(('just_random_taken_Cp_scaled', str(round(percent_random, 2)) + "%"))
        #################################################



        # ############################## FOROUGH ALGORITHM ###################
        # matrix = complex_network.adj_matrix()
        # percent_forough = network.control.percentage_control_forough(matrix, n, self.driver_nodes)[0] * 100
        # output += "  >>Forough algorithm %%%f\n" % percent_forough
        # ############################## FOROUGH ALGORITHM ###################

        result_matrix.append(("actual_driver_node_Cp", str(percent_algorithmic * 100) + "%"))

        # distribution_criteria = {
        #     "in_degree_distribution": self.complex_network.graph.in_degree(),
        #     "in_degree_criteria": distribution_criteria["in_degree_criteria"],
        #     "out_degree_distribution": self.complex_network.graph.out_degree(),
        #     "out_degree_criteria": distribution_criteria["out_degree_criteria"],
        #     "total_degree_distribution": self.complex_network.graph.degree(),
        #     "total_degree_criteria": distribution_criteria["total_degree_criteria"]
        # }

        distribution_criteria = {
            distribution_type + '_degree_distribution': self.complex_network.graph.in_degree(),
            distribution_type + "_degree_criteria": distribution_criteria[distribution_type + "_degree_criteria"]
        }

        # distribution_criteria[distribution_type + '_degree_distribution'] = self.complex_network.graph.in_degree()
        #
        # distribution_criteria[distribution_type + "_degree_criteria"] = \
        #     distribution_criteria[distribution_type + "_degree_criteria"]

        randomly_chosen_driver_nodes = self._heuristic(
            actual_percentage_control=percent_algorithmic,
            number_of_buckets_ie_tiles=4,  # fix this later
            expected_number_of_driver_nodes=expected_number_of_driver_nodes,
            degree_distribution_dict=distribution_criteria[distribution_type + "_degree_distribution"],
            criteria=distribution_criteria[distribution_type + "_degree_criteria"],
            max_samples_number=max_samples_number
        )

        percent_heuristic = self.complex_network.control.get_percentage_control(
            randomly_chosen_driver_nodes['driver_nodes'],
            longest_shortest_path=use_longest_shortest_path,
            scale=percent_algorithmic
        ) * 100

        number_of_sampled_nodes_intersect_with_one_mds_driver_nodes = 0
        for node in randomly_chosen_driver_nodes['driver_nodes']:
            if node in self.driver_nodes:
                number_of_sampled_nodes_intersect_with_one_mds_driver_nodes += 1

        # output += ("\n@Heuristic [%s-degree] Control Percentage: %f%%\n" %
        #            (distribution_type, round(percent_heuristic, 2)))
        result_matrix.append(("H_Cp_scaled", str(round(percent_heuristic, 2)) + "%"))

        # output += "-->Expected_number_of_driver_nodes: %d\n" % expected_number_of_driver_nodes
        result_matrix.append(("H_expected_number_of_driver_nodes", expected_number_of_driver_nodes))

        result_matrix.append((
            "H_Cp_diff_actual_scaled",
            # str(round(percent_algorithmic * 100 - percent_heuristic, 3)) + "%"))
            str(round(100 - percent_heuristic, 3)) + "%"))
        result_matrix.append(
            ("H_Cp_diff_random_scaled",
             # str(round(percent_algorithmic * 100 - percent_random, 3)) + "%"))
             str(round(100 - percent_random, 3)) + "%"))

        result_matrix.append(
            (
                "H_intersect_actual_driver_nodes",
                "%s%%" % str(
                    round(number_of_sampled_nodes_intersect_with_one_mds_driver_nodes / number_of_driver_nodes,
                          2) * 100)[:5]
            ))

        # output += "-->Intersect with actual driver nodes:%s%%\n" % str(round(
        #     (number_of_sampled_nodes_intersect_with_driver_nodes / number_of_driver_nodes), 2) * 100)

        # output += "-->Percentage of sampled nodes:%s\n" % \
        #           str(round(randomly_chosen_driver_nodes['number_of_sampled_nodes'] / n, 2) * 100)
        result_matrix.append(
            (
                "H_percentage_of_sampled_nodes",
                "%s%%" % str(round(randomly_chosen_driver_nodes['number_of_sampled_nodes'] / n, 2) * 100)[:5]
            )
        )

        # output += "\n@%s\n" % distribution_criteria[distribution_type + "_degree_criteria"]
        result_matrix.append((
            "H_buckets_size",
            distribution_criteria[distribution_type + "_degree_criteria"]['bucket_size']))
        result_matrix.append((
            "H_buckets_ranges",
            distribution_criteria[distribution_type + "_degree_criteria"]['range']))

        buckets_len = [len(item) for item in randomly_chosen_driver_nodes['buckets']]

        # output += "-->Number of nodes in each bucket: %s" % str(buckets_len)
        result_matrix.append((
            'H_number_of_nodes_in_buckets',
            str(buckets_len)
        ))

        result_matrix.append((
            'H_percent_#_driver_/_expected',
            "%s%%" % str(round(numpy.sum(buckets_len) / expected_number_of_driver_nodes, 2) * 100)
        ))

        # output += "-->Chosen driver nodes in each bucket: \n  -->%s" % randomly_chosen_driver_nodes['buckets']
        result_matrix.append((
            'H_driver_nodes_in_each_bucket',
            randomly_chosen_driver_nodes['buckets']))

        # percent = network.control.get_percentage_control(
        #     randomly_chosen_driver_nodes[1],
        #     logest_shortest_path=logest_shortest_path)
        # print("@Heuristic chosen %d node Control Percentage Sort: %%%f" %
        #       (len(randomly_chosen_driver_nodes[1), percent))

        # print(randomly_chosen_driver_nodes[0])
        # print(randomly_chosen_driver_nodes[1])
        # print(randomly_chosen_driver_nodes)

        return {
            'message': '',  # output
            'result_matrix': result_matrix,
            'result_dict': dict([(a[0], a[1]) for a in result_matrix])
        }

    def _heuristic(
            self,
            actual_percentage_control,
            number_of_buckets_ie_tiles,
            expected_number_of_driver_nodes,
            degree_distribution_dict,
            criteria,
            max_samples_number
    ):

        graph = self.complex_network.graph

        graph_nodes = graph.nodes()

        if len(graph_nodes) == 0:
            raise Exception('[complex_network.py sample_nodes_heuristic] len(graph.nodes() cannot be 0')

        if expected_number_of_driver_nodes > len(graph_nodes):
            return "number of requested driver nodes is bigger than the number of nodes in the graph."

        buckets_ie_tiles_contain_nodes = dict([(i, []) for i in range(1, number_of_buckets_ie_tiles + 1)])
        successful_sampled_nodes_count = 0
        buckets_size = dict(
            [(bucket_size['tile'], np.ceil(bucket_size['size'] * expected_number_of_driver_nodes)) for bucket_size in
             criteria['bucket_size']])
        buckets_ranges = dict([(i, criteria['range'][i - 1]) for i in range(1, len(criteria['range']) + 1)])

        self.event_buckets_ranges_created({'buckets_ranges': buckets_ranges})

        number_of_sampled_nodes = 0
        nodes_in_any_bucket = []

        while successful_sampled_nodes_count < expected_number_of_driver_nodes:
            if max_samples_number > 0 and max_samples_number < number_of_sampled_nodes:
                break

            # sampled all nodes, could not satisfy the requested # of driver nodes
            if len(graph_nodes) == 0:
                break

            node = random.choice(graph_nodes)
            number_of_sampled_nodes += 1
            node_degree = degree_distribution_dict[node]

            sample_node_post_parameters = {
                'node': node,
                'bucket_number': None,
                'successful_sampled_nodes_count': None,
                'buckets': None,
                'number_of_sampled_nodes': None,
                'nodes_in_any_bucket': None,
                'actual_percentage_control': actual_percentage_control
            }

            if node not in nodes_in_any_bucket:

                graph_nodes.remove(node)
                # first identify to which bucket the node belong to.
                # second if the bucket is not full add it to the bucket

                for bucket_number in buckets_ranges.keys():
                    if len(buckets_ie_tiles_contain_nodes[bucket_number]) >= buckets_size[bucket_number]:
                        continue

                    if buckets_ranges[bucket_number][0] <= node_degree <= buckets_ranges[bucket_number][1]:
                        nodes_in_any_bucket.append(node)
                        buckets_ie_tiles_contain_nodes[bucket_number].append(node)
                        successful_sampled_nodes_count += 1

                        # call event handlers
                        sample_node_post_parameters['bucket_number'] = bucket_number

            sample_node_post_parameters['buckets'] = buckets_ie_tiles_contain_nodes
            sample_node_post_parameters['successful_sampled_nodes_count'] = successful_sampled_nodes_count
            sample_node_post_parameters['number_of_sampled_nodes'] = number_of_sampled_nodes
            sample_node_post_parameters['nodes_in_any_bucket'] = nodes_in_any_bucket

            self.event_sample_node_post(sample_node_post_parameters)

        all_buckets = list([value for key, value in buckets_ie_tiles_contain_nodes.items()])

        return {
            'buckets': all_buckets,
            'driver_nodes': list(np.concatenate(all_buckets, axis=0)),
            'number_of_sampled_nodes': number_of_sampled_nodes
        }


def handle_foo(sender, is_driver_node):
    print(sender)
    print(is_driver_node)


if __name__ == '__main__':
    pass
