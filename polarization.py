import numpy as np
import networkx as nx
import random
import itertools
from complex_networks.general_tools import GeneralTools
import complex_networks.tools as tools
import sys
import os.path


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


def fx(edu1, edu2, econ1, econ2, c, d):
    # c = 0.01  # weakens the affect
    # c = 0.5
    # d = 1

    def f(a, b):
        return (1.0 / d) * (
            (c / (c + 3.0)) + (np.absolute(a + b) / 6.0) * (
                (c / (c + np.abs(a - b))) - (c / (c + 3.0)))
        )

    eco_dif = np.abs(econ1 - econ2)
    edu_dif = np.abs(edu1 - edu2)
    f_tot = f(edu1, edu2) * f(econ1, econ2)
    if f_tot > 1:
        f_tot = 1

    bin = np.random.binomial(1, f_tot)

    if bin == 1:
        return True
    else:
        return False
    pass


def using_permutation(n, target_k, c, d, x_cor_max_min, y_cor_max_min, links_weight_std,
                      education_mean, education_std, economic_mean, economic_std):
    G = nx.DiGraph()

    if education_std < 0.4 or economic_std < 0.4 or links_weight_std < 0.4:
        # raise Exception("none of the std ies can be < 0.4")
        pass

    for i in range(0, n):
        G.add_node(i)

        def generate_education_economic(mean, std):
            val = np.round(np.random.normal(mean, std), 3)
            if val < 0:
                return generate_education_economic(mean, std)
            if val > 3:
                val = 2.999
            return val
            pass

        education = generate_education_economic(education_mean, education_std)
        economic = generate_education_economic(economic_mean, economic_std)
        G.node[i] = {
            # "education": np.random.uniform(0, 3.01),
            # "economic": np.random.uniform(0, 3),
            "color": 105,  # blue
            "driver": 0,
            "provider": 0,
            "education": education,
            "economic": economic,
            "WHO": str(i),
            "XCOR": np.round(np.random.uniform(-1.0 * x_cor_max_min, x_cor_max_min), 3),
            "YCOR": np.round(np.random.uniform(-1.0 * y_cor_max_min, y_cor_max_min), 3),
        }

    edges = list(itertools.permutations(range(n), 2))
    random.shuffle(edges)
    edges_lenght = len(edges)

    # for edge in edges:
    while True:
        if (2.0 * G.number_of_edges()) / n >= target_k:
            break

        if edges_lenght == 0:
            break

        edge = edges.pop()
        edges_lenght -= 1
        n1 = edge[0]
        n2 = edge[1]

        if n1 == n2:
            continue
        # education_diff = np.absolute(G.node[n1]["education"] - G.node[n2]["education"])
        # economic_diff = np.absolute(G.node[n1]["economic"] - G.node[n2]["economic"])

        education1 = G.node[n1]["education"]
        education2 = G.node[n2]["education"]
        economic1 = G.node[n1]["economic"]
        economic2 = G.node[n2]["economic"]

        # education_uniform = np.random.uniform(0, education_diff)
        # economic_uniform = np.random.uniform(0, economic_diff)

        # if education_uniform < 0 or economic_uniform < 0:
        #     raise Exception("education_uniform = {} or economic_uniform = {} cannot be < 0 ".format(
        #         education_uniform, economic_uniform))

        # prob_education = np.exp(-1 * education_uniform)
        # prob_economic = np.exp(-1 * economic_uniform)
        #
        # if prob_education < 0 or prob_economic < 0:
        #     raise Exception("prob_education = {} or prob_economic = {} cannot be < 0".format(
        #         prob_education, prob_economic))

        # if prob_education >= prob_cut_off and prob_economic >= prob_cut_off:

        if fx(edu1=education1, edu2=education2, econ1=economic1, econ2=economic2, c=c, d=d):
            weight = np.absolute(np.round(np.random.normal(0, links_weight_std), 3))
            if weight >= 1:
                weight = 1.0

            G.add_edge(n2, n1, {
                "weight": weight,
                "label": str(weight),
                "color": 55  # green
            })

    return G


def custom_graph():
    G = nx.DiGraph()
    G.add_node(0, {"education": 0, "economic": 0, "WHO": 0, "XCOR": 3, "YCOR": 3})
    G.add_node(1, {"education": 0, "economic": 0, "WHO": 1, "XCOR": -3, "YCOR": 3})
    G.add_node(2, {"education": 0, "economic": 0, "WHO": 2, "XCOR": -3, "YCOR": -3})
    G.add_node(3, {"education": 0, "economic": 0, "WHO": 3, "XCOR": 3, "YCOR": -3})

    G.add_edge(0, 1, weight=0.1, label=0.1)
    G.add_edge(0, 2, weight=0.1, label=0.1)
    G.add_edge(1, 3, weight=0.1, label=0.1)
    G.add_edge(2, 3, weight=0.1, label=0.1)
    G.add_edge(3, 0, weight=0.1, label=0.1)
    return G


def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
    pass


def add_news_provider_connect_to_driver_nodes_in_their_quadrants(G, driver_node, location_range):
    new_node_id = G.number_of_nodes()
    G.add_node(new_node_id, {
        "education": -1, "economic": -1, "WHO": new_node_id, "color": "29",
        "driver": 0, "provider": 1,
        "XCOR": np.random.uniform(location_range, -1 * location_range),
        "YCOR": np.random.uniform(location_range, location_range + 1)})
    weight = round(np.random.uniform(0.4, 0.9), 3)
    G.add_edge(new_node_id, driver_node, weight=weight, label=weight)
    pass


def add_four_news_provider_to_mds(G, mds, max_num_subscribers):
    connected_driver_nodes = []

    def add_provider_and_link_to_subscriber(xcor, ycor, max_num_subs):
        new_node_id = G.number_of_nodes()
        G.add_node(new_node_id, {
            "education": -1, "economic": -1, "WHO": new_node_id, "color": "29",
            "driver": 0, "provider": 1,
            "XCOR": xcor,
            "YCOR": ycor})

        weight = 0.8

        eligible_news_subscribers = []

        for driver_node_2 in mds:
            driver_node_xcor = G.node[driver_node_2]["XCOR"]
            driver_node_ycor = G.node[driver_node_2]["YCOR"]

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
                           weight=weight,
                           label=weight,
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
            "driver": 0, "provider": 1.0,
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

            node_xcor = G.node[node]["XCOR"]
            node_ycor = G.node[node]["YCOR"]

            distance = np.sqrt(
                np.power(xcor - node_xcor, 2) + np.power(ycor - node_ycor, 2)) / inverse_distance_divide_by

            if 1 - distance <= 0.01:
                bino = 0
            else:
                bino = np.random.binomial(1, 1.0 - distance)

            if bino == 1:
                connected_driver_nodes.append(node)
                G.add_edge(new_node_id, node,
                           weight=weight,
                           label=weight,
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
            "education": -1.0, "economic": -1.0, "WHO": new_node_id, "color": "29",
            "driver": 0, "provider": 1.0,
            "XCOR": xcor,
            "YCOR": ycor})

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

            node_xcor = G.node[node]["XCOR"]
            node_ycor = G.node[node]["YCOR"]

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
                           weight=weight,
                           label=weight,
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


def stats(G):
    result = ""
    check_cutoff = 2.5
    edu = []
    for node in G.nodes():
        edu.append(G.node[node]["education"])
        # if G.node[node]["education"] > 2:
        #     econ.add(node)
        #     pass

    edu_condition = []
    weights = []
    ppl_edu = []

    for node in G.nodes():
        ppl_edu.append(G.node[node]["education"])
        pass

    result += "ppl edu > 2.5 = {} ".format(len([q for q in ppl_edu if q > 2.5]))

    edu_diffs = []
    econ_diffs = []
    for edge in G.edges():
        weights.append(G.edge[edge[0]][edge[1]]["weight"])
        edu_diffs.append(np.absolute(G.node[edge[0]]["education"] - G.node[edge[1]]["education"]))
        econ_diffs.append(np.absolute(G.node[edge[0]]["economic"] - G.node[edge[1]]["economic"]))
        if G.node[edge[0]]["education"] > check_cutoff and G.node[edge[1]]["education"] > check_cutoff:
            edu_condition.append(np.absolute(G.node[edge[0]]["education"] - G.node[edge[1]]["education"]))
            pass

    result += "$avg edu = {} $ num-edge-with-both-nodes-edu > {} = {} $avg-edu-diff = {} $avg-econ-diff = {} $<k> = {}".format(
        round(np.average(edu), 3), check_cutoff, len(edu_condition),
        round(np.average(edu_diffs), 3),
        round(np.average(econ_diffs), 3),
        round((G.number_of_edges() * 2.0) / G.number_of_nodes(), 3))

    result += " $weights-avg = {} $count-weights > 0.5 = {}".format(
        round(np.average(weights), 3),
        len([w for w in weights if w >= 0.5])
    )

    return result


if __name__ == "__main__":
    debug = True
    if len(sys.argv) > 2:
        n = int(sys.argv[1])
        target_k = float(sys.argv[2])
        x_cor_max_min = float(sys.argv[3])
        y_cor_max_min = float(sys.argv[4])
        links_weight_std = float(sys.argv[5])  # links weight between 0 0.2
        education_mean = float(sys.argv[6])
        education_std = float(sys.argv[7])
        economic_mean = float(sys.argv[8])
        economic_std = float(sys.argv[9])  # the smaller the less economic
        c = float(sys.argv[10])
        d = float(sys.argv[11])
        debug = bool(int(sys.argv[12]))
        run_number = int(float(sys.argv[13]))
        max_num_subscribers = int(float(sys.argv[14]))
        subscription_weight_max_uniform = float(sys.argv[15])
        subscribers_inverse_distance_factor = float(sys.argv[16])
        pass

    if debug is True:
        n = 100
        target_k = 5.0
        # prob_cut_off = 0.940  #
        x_cor_max_min = 6.0  # control location belief
        y_cor_max_min = 6.0  # control location socio-econ belief etc
        links_weight_std = 0.09  # links weight between 0 0.2
        education_mean = 1.5
        education_std = 0.55
        economic_mean = 1.5
        economic_std = 0.55  # the smaller the less economic
        c = 1
        d = 1
        run_number = 10
        max_num_subscribers = 0
        subscription_weight_max_uniform = 0.3
        subscribers_inverse_distance_factor = 9.0
        pass

    G = using_permutation(
        n=n, target_k=target_k,
        c=c, d=d, x_cor_max_min=x_cor_max_min, y_cor_max_min=y_cor_max_min,
        links_weight_std=links_weight_std, education_mean=education_mean, education_std=education_std,
        economic_mean=economic_mean, economic_std=economic_std)

    mds = GeneralTools.identify_driver_nodes(
        networkx_digraph=G, root_folder_work="pol", debug=False, draw_graphs=False,
        show_plots=False, network_id=1)

    stats_val = stats(G)
    # tools.clipboard_copy(stats_val)
    if debug:
        print(stats_val)
        pass
    # n target_k x_cor_max_min y_cor_max_min links_weight_std education_mean education_std economic_mean economic_std c d debug run_number
    # 100 5 6 6 0.09 1.5 0.55 1.5 0.55 0.5 1.0 1 10
    for driver_node in mds:
        # add_news_provider(G, mds, location_range=-7)
        G.node[driver_node]["driver"] = 1

    # add_four_news_provider_no_mds(
    #     G, max_num_subscribers=max_num_subscribers,
    #     weight_max_uniform=subscription_weight_max_uniform,
    #     inverse_distance_divide_by=subscribers_inverse_distance_factor)

    # max_num_subscribers = 4
    add_fixed_providers_no_mds(
        G, max_num_subscribers=max_num_subscribers,
        weight_max_uniform=subscription_weight_max_uniform,
        inverse_distance_factor=subscribers_inverse_distance_factor)

    i = 0
    while True:
        path = "d:\\temp\\pol\\exp\\{}_pol_{}.gml".format(run_number, i)
        if os.path.isfile(path) is False:
            break
            pass
        i += 1

    # print (path)

    sys.stdout.write(path)
    nx.write_gml(G, path)

    line_prepender(
        path,
        "#n = {0}\n#target_k = {1}\n#c = {2}\n#d = {10}"
        "\n#x_cor_max_min = {3}\n#y_cor_max_min = {4}\n#links_weight_std = {5}\n#education_mean = {6}"
        "\n#education_std = {7}\n#economic_mean = {8}\n#economic_std = {9}\n#max_num_subscribers = {12}"
        "\n#subscription_weight_max_uniform = {13}"
        "\n#subscribers_inverse_distance_factor = {14}\n#stat = {11}".format(
            n, target_k, c, x_cor_max_min, y_cor_max_min, links_weight_std, education_mean, education_std,
            economic_mean, economic_std, d, stats_val, max_num_subscribers, subscription_weight_max_uniform,
            subscribers_inverse_distance_factor
        ))

    pass
