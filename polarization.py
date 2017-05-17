import numpy as np
import networkx as nx
import random
import itertools
from complex_networks.general_tools import GeneralTools
import complex_networks.constants as constants


def get_prob(x):
    return np.exp(-0.9 * x)


def using_permutation(n, target_k, prob_cut_off, x_cor_max_min, y_cor_max_min, links_weight_std,
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
            "color": 105, # blue
            "driver": 0,
            "provider": 0,
            "education": education,
            "economic": economic,
            "WHO": str(i),
            "XCOR": np.round(np.random.uniform(-1 * x_cor_max_min, x_cor_max_min), 3),
            "YCOR": np.round(np.random.uniform(-1 * y_cor_max_min, y_cor_max_min), 3),
        }

    edges = list(itertools.permutations(range(n), 2))
    random.shuffle(edges)
    edges_lenght = len(edges)

    # for edge in edges:
    while True:
        if (2 * G.number_of_edges()) / n >= target_k:
            break

        if edges_lenght == 0:
            break

        edge = edges.pop()
        edges_lenght -= 1
        n1 = edge[0]
        n2 = edge[1]

        if n1 == n2:
            continue
        education_diff = np.absolute(G.node[n1]["education"] - G.node[n2]["education"])
        economic_diff = np.absolute(G.node[n1]["economic"] - G.node[n2]["economic"])

        education1 = G.node[n1]["education"]
        education2 = G.node[n2]["education"]

        education_uniform = np.random.uniform(0, education_diff)
        economic_uniform = np.random.uniform(0, economic_diff)

        if education_uniform < 0 or economic_uniform < 0:
            raise Exception("education_uniform = {} or economic_uniform = {} cannot be < 0 ".format(
                education_uniform, economic_uniform))

        if education1 > 2.0 and education2 > 2.0:
            pass
        elif education1 <= 2.0 and education2 <= 2.0:
            pass
        else:
            pass

        prob_education = np.exp(-1 * education_uniform)
        prob_economic = np.exp(-1 * economic_uniform)

        if prob_education < 0 or prob_economic < 0:
            raise Exception("prob_education = {} or prob_economic = {} cannot be < 0".format(
                prob_education, prob_economic))

        if prob_education >= prob_cut_off and prob_economic >= prob_cut_off:
            # weight = np.round(np.random.uniform(0, links_weight_std), 2)
            weight = np.absolute(np.round(np.random.normal(0, links_weight_std), 3))
            if weight > 1:
                weight = 0.999

            G.add_edge(n2, n1, {
                "weight": weight,
                "label": str(weight)
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


if __name__ == "__main__":

    n = 200
    target_k = 2
    prob_cut_off = 0.900
    x_cor_max_min = 6
    y_cor_max_min = 6
    links_weight_std = 0.15  # links wieght between 0 0.2
    education_mean = 1.5
    education_std = 0.8
    economic_mean = 1.5
    economic_std = 0.8  # the smaller the less economic

    G = using_permutation(
        n=n, target_k=target_k,
        prob_cut_off=prob_cut_off, x_cor_max_min=x_cor_max_min, y_cor_max_min=y_cor_max_min,
        links_weight_std=links_weight_std, education_mean=education_mean, education_std=education_std,
        economic_mean=economic_mean, economic_std=economic_std)

    mds = GeneralTools.identify_driver_nodes(
        networkx_digraph=G, root_folder_work="pol", debug=False, draw_graphs=False,
        show_plots=False, network_id=1)

    def add_news_provider(G, driver_node, location_range):
        new_node_id = G.number_of_nodes()
        G.add_node(new_node_id, {
            "education": -1, "economic": -1, "WHO": new_node_id, "color": "29",
            "driver": 0, "provider": 1,
            "XCOR": np.random.uniform(location_range, -1 * location_range),
            "YCOR": np.random.uniform(location_range, location_range + 1)})
        G.add_edge(new_node_id, driver_node, weight=1, label=1)
        pass


    for driver_node in mds:
        add_news_provider(G, driver_node, location_range=-7)
        G.node[driver_node]["driver"] = 1

    edu = []
    for node in G.nodes():
        edu.append(G.node[node]["education"])
        # if G.node[node]["education"] > 2:
        #     econ.add(node)
        #     pass

    edu_condition = []
    for edge in G.edges():
        if G.node[edge[0]]["education"] > 2.5 and G.node[edge[1]]["education"] > 2.5:
            edu_condition.append(np.absolute(G.node[edge[0]]["education"] - G.node[edge[1]]["education"]))
            pass
            # val = np.absolute(G.node[node]["education"] - G.node[n]["education"])
            # if val > 1:
            #     edu.append(val)
            #     break

    print (
        "avg edu : {} #edge with both nodes edu > 2 : {} <k> = {}".format(
            np.average(edu), len(edu_condition), (G.number_of_edges() * 2.0) / G.number_of_nodes()))

    path = "d:\\temp\\pol\\" + "pol1" + ".gml"
    nx.write_gml(G, path)

    line_prepender(
        path,
        "#n = {0}\n#target_k = {1}\n#prob_cut_off = {2}\n"
        "#x_cor_max_min = {3}\n#y_cor_max_min = {4}\n#links_weight_std = {5}\n#education_mean = {6}\n#education_std = {7}"
        "\n#economic_mean = {8}\n#economic_std = {9}\n".format(
            n, target_k, prob_cut_off, x_cor_max_min, y_cor_max_min, links_weight_std, education_mean, education_std,
            economic_mean, economic_std
        ))
