from complex_networks.general_tools import GeneralTools
import complex_networks.tools as tools
from pandas import *
import math
import numpy as np
import networkx as nx
import random
import itertools
import sys
import os.path

def add_four_news_provider_to_mds(G, mds, max_num_subscribers):
    connected_driver_nodes = []

    def add_provider_and_link_to_subscriber(xcor, ycor, max_num_subs):
        new_node_id = G.number_of_nodes()
        G.add_node(new_node_id, {
            "education": -1, "economic": -1, "WHO": new_node_id, "color": "29",
            "driver": 0, "isProvider": 1,
            "XCOR": xcor,
            "YCOR": ycor})

        weight = 0.8

        eligible_news_subscribers = []

        for driver_node_2 in mds:
            driver_node_xcor = float(G.node[driver_node_2]["XCOR"])
            driver_node_ycor = float(G.node[driver_node_2]["YCOR"])

            if xcor > 0 and ycor > 0 and driver_node_xcor > 0 and driver_node_ycor > 0:
                eligible_news_subscribers.append(driver_node_2)
            elif xcor < 0 and ycor > 0 and driver_node_xcor < 0 and driver_node_ycor > 0:
                eligible_news_subscribers.append(driver_node_2)
            elif xcor < 0 and ycor < 0 and driver_node_xcor < 0 and driver_node_ycor < 0:
                eligible_news_subscribers.append(driver_node_2)
            elif xcor > 0 and ycor < 0 and driver_node_xcor > 0 and driver_node_ycor < 0:
                eligible_news_subscribers.append(driver_node_2)

        if max_num_subs == 0:
            max_num_subs = len(eligible_news_subscribers)

        for i in range(0, max_num_subs):
            if len(eligible_news_subscribers) > 0:
                random.shuffle(eligible_news_subscribers)
                node_to_subscriber = eligible_news_subscribers.pop()
                connected_driver_nodes.append(node_to_subscriber)
                G.add_edge(new_node_id, node_to_subscriber,
                           weight=str(weight),
                           label=str(weight),
                           color="9.9"  # white
                           )
                pass
        pass

    pos = 9
    add_provider_and_link_to_subscriber(pos, pos, max_num_subscribers)
    add_provider_and_link_to_subscriber(-1 * pos, pos, max_num_subscribers)
    add_provider_and_link_to_subscriber(-1 * pos, -1 * pos, max_num_subscribers)
    add_provider_and_link_to_subscriber(pos, -1 * pos, max_num_subscribers)


def add_four_news_provider_no_mds(G, max_num_subscribers, weight_max_uniform, inverse_distance_divide_by):
    connected_driver_nodes = []
    added_providers_nodes = []

    def add_provider_and_link_to_subscriber(xcor, ycor, max_num_subs):

        new_node_id = G.number_of_nodes()
        added_providers_nodes.append(new_node_id)
        G.add_node(new_node_id, {
            "education": -1.0, "economic": -1.0, "WHO": new_node_id, "color": "29",
            "driver": 0, "isProvider": 1.0,
            "XCOR": xcor,
            "YCOR": ycor})

        # weight = np.absolute(np.round(np.random.normal(0, weight_std), 3))
        weight = np.round(np.random.uniform(0.1, weight_max_uniform), 3)

        nodes_shuffled = G.nodes()[:]
        random.shuffle(nodes_shuffled)
        for node in nodes_shuffled:
            if node == new_node_id or node in added_providers_nodes:  # avoid self-link
                continue

            if max_num_subs > 0 and len(connected_driver_nodes) >= max_num_subs:
                return

            node_xcor = float(G.node[node]["XCOR"])
            node_ycor = float(G.node[node]["YCOR"])

            distance = np.sqrt(
                np.power(xcor - node_xcor, 2) + np.power(ycor - node_ycor, 2)) / inverse_distance_divide_by

            if 1 - distance <= 0.01:
                bino = 0
            else:
                bino = np.random.binomial(1, 1.0 - distance)

            if bino == 1:
                connected_driver_nodes.append(node)
                G.add_edge(new_node_id, node,
                           weight=str(weight),
                           label=str(weight),
                           color="9.9"  # white
                           )
                pass
        pass

    pos = 9
    add_provider_and_link_to_subscriber(pos, pos, max_num_subscribers)
    add_provider_and_link_to_subscriber(-1.0 * pos, pos, max_num_subscribers)
    add_provider_and_link_to_subscriber(-1.0 * pos, -1.0 * pos, max_num_subscribers)
    add_provider_and_link_to_subscriber(pos, -1.0 * pos, max_num_subscribers)


def add_fixed_providers_no_mds(G, max_num_subscribers, weight_max_uniform, inverse_distance_factor):
    connected_driver_nodes = []
    added_providers_nodes = []

    def add_provider_and_link_to_subscriber(xcor, ycor, max_num_subs):

        added_links = 0

        new_node_id = G.number_of_nodes()
        added_providers_nodes.append(new_node_id)
        G.add_node(new_node_id, {
            "education": "-1.0", "economic": "-1.0",
            "WHO": new_node_id, "color": "29",
            "driver": "0.0", "isProvider": "1.0",
            # "colourList": "",
            "XCOR": str(xcor),
            "YCOR": str(ycor)})

        # weight = np.absolute(np.round(np.random.normal(0, weight_std), 3))
        weight = np.round(np.random.uniform(0.1, weight_max_uniform), 3)

        nodes_shuffled = G.nodes()[:]
        random.shuffle(nodes_shuffled)
        for node in nodes_shuffled:
            if node == new_node_id or node in added_providers_nodes:  # avoid self-link
                continue

            if node == new_node_id:  # avoid self-link
                continue
            # if max_num_subs > 0 and len(connected_driver_nodes) >= max_num_subs:
            #     return
            if max_num_subs > 0 and added_links >= max_num_subs:
                return

            node_xcor = float(G.node[node]["XCOR"])
            node_ycor = float(G.node[node]["YCOR"])

            # ;;;;;;;;;;;;;;; no need for randomness the nodes are already shuffled ;;;;;;
            # distance = np.sqrt(
            #     np.power(xcor - node_xcor, 2) + np.power(ycor - node_ycor, 2)) \
            #            / inverse_distance_divide_by
            #
            # if 1 - distance <= 0.01:
            #     bino = 0
            # else:
            #     bino = np.random.binomial(1, 1.0 - distance)

            distance = np.sqrt(np.power(xcor - node_xcor, 2) + np.power(ycor - node_ycor, 2))

            # if bino == 1:
            if distance <= inverse_distance_factor:
                connected_driver_nodes.append(node)
                G.add_edge(new_node_id, node,
                           weight=str(weight),
                           label=str(weight),
                           color="9.9"  # white
                           )
                added_links += 1
                pass
        pass

    add_provider_and_link_to_subscriber(4.2, 9.8, max_num_subscribers)  # cnn
    add_provider_and_link_to_subscriber(9.7, 9.5, max_num_subscribers)  # cnn

    add_provider_and_link_to_subscriber(-9.8, 9.7, max_num_subscribers)  # bbc
    add_provider_and_link_to_subscriber(-8.8, 3.6, max_num_subscribers)

    add_provider_and_link_to_subscriber(-9.61, -9.92, max_num_subscribers)  # fox
    add_provider_and_link_to_subscriber(-6.532, -8.8, max_num_subscribers)

    add_provider_and_link_to_subscriber(9.8, -8.8, max_num_subscribers)
    add_provider_and_link_to_subscriber(6.6, -9.5, max_num_subscribers)


def add_news_provider_connect_to_driver_nodes_in_their_quadrants(G, driver_node, location_range):
    new_node_id = G.number_of_nodes()
    G.add_node(new_node_id, {
        "education": -1, "economic": -1, "WHO": new_node_id, "color": "29",
        "driver": 0, "isProvider": 1,
        "XCOR": np.random.uniform(location_range, -1 * location_range),
        "YCOR": np.random.uniform(location_range, location_range + 1)})
    weight = round(np.random.uniform(0.4, 0.9), 3)
    G.add_edge(new_node_id, driver_node, weight=str(weight), label=str(weight))
    pass


def custom_graph():
    G = nx.DiGraph()
    G.add_node(0, {"education": 0, "economic": 0, "WHO": 0, "XCOR": 3, "YCOR": 3})
    G.add_node(1, {"education": 0, "economic": 0, "WHO": 1, "XCOR": -3, "YCOR": 3})
    G.add_node(2, {"education": 0, "economic": 0, "WHO": 2, "XCOR": -3, "YCOR": -3})
    G.add_node(3, {"education": 0, "economic": 0, "WHO": 3, "XCOR": 3, "YCOR": -3})

    G.add_edge(0, 1, weight="0.1", label="0.1")
    G.add_edge(0, 2, weight="0.1", label="0.1")
    G.add_edge(1, 3, weight="0.1", label="0.1")
    G.add_edge(2, 3, weight="0.1", label="0.1")
    G.add_edge(3, 0, weight="0.1", label="0.1")
    return G


def fx_old(a, b, h=0, c=0, d=0):
    h = 5
    c = 0.3
    d = 1

    func_val = (c / d) * (1 - np.exp(
        1 / (-1 * h * np.absolute(3 - ((a + b) / 2))))) * \
               (1 / (c + np.absolute(a - b)))

    uni = np.random.uniform(0, 1.0 / func_val)
    bin = np.random.binomial(1, func_val)
    if bin == 1:
        return True
    else:
        return False
    pass


def homophily_matrix(G, lbl, save_p):
    nums = [[0 for x in range(30)] for y in range(30)]

    for edge in G.edges():
        from_edu = float(G.node[edge[0]][lbl])
        to_edu = float(G.node[edge[1]][lbl])

        i = int((math.floor((from_edu - 0.00000001) * 10) / 10) * 10)
        j = int((math.floor((to_edu - 0.00000001) * 10) / 10) * 10)

        nums[i][j] += 1

        pass

    df = DataFrame(nums)
    df.rename(columns=lambda x: x * 0.1, inplace=True)
    df.rename(dict(zip(df.index.tolist(), [round(x * 0.1, 1) for x in df.index.tolist()])), inplace=True)

    df.to_csv(save_p)

    pass
