from pandas import *
import math
import numpy as np
import networkx as nx
import random
import itertools
import sys
import os.path


def fx(edu1, edu2, econ1, econ2, c, d):
    # c = 0.01  # weakens the affect
    # c = 0.5
    # d = 1

    def f(a, b):
        return c / (c + np.power(a - b, 2))

    # def f(a, b):
    #     return (1.0 / d) * (
    #         (c / (c + 3.0)) + (np.absolute(a + b) / 6.0) * (
    #             (c / (c + np.abs(a - b))) - (c / (c + 3.0)))
    #     )

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
                      education_mean, education_std, economic_mean, economic_std, edu_econ_correlation):
    def position_news_consumer(economic_level, edu_level):
        # return xcor, ycor
        # XCOR: edu_level is attr-var ---map-to---> social-belief is state-var
        # YCOR: socio_econ_level is attr-var ---map-to---> fiscal-view is state-var

        if economic_level > 0 and edu_level > 0:
            return np.random.uniform(0, x_cor_max_min), np.random.uniform(0, y_cor_max_min)
        elif economic_level < 0 and edu_level > 0:
            return np.random.uniform(0, 1 * x_cor_max_min), np.random.uniform(0, -1 * y_cor_max_min)
        elif economic_level < 0 and edu_level < 0:
            return np.random.uniform(0, -1 * x_cor_max_min), np.random.uniform(0, -1 * y_cor_max_min)
        else:
            return np.random.uniform(0, -1 * x_cor_max_min), np.random.uniform(0, 1 * y_cor_max_min)
        pass

    def generate_education_economic(mean, std, edu_econ_correlation_val):
        # return np.round(np.random.uniform(-2, 2), 3), \
        #        np.round(np.random.uniform(-2, 2), 3)

        mean = np.random.choice([-1, 1], p=[0.5, 0.5])
        std = 0.7
        # edu_val = np.round(np.random.normal(mean, std), 3)

        edu_val = np.round(np.random.uniform(-2, 2), 3)

        if edu_econ_correlation_val == 0:
            econ_val = np.round(np.random.uniform(-2, 2), 3)
        else:
            econ_val = np.round(np.random.normal(edu_val, edu_econ_correlation_val), 3)
            pass

        if edu_val < -2:
            edu_val = -2
        if edu_val > 2:
            edu_val = 2
        if econ_val < -2:
            econ_val = -2
        if econ_val > 2:
            econ_val = 2
        return edu_val, econ_val
        pass

    G = nx.DiGraph()

    if education_std < 0.4 or economic_std < 0.4 or links_weight_std < 0.4:
        # raise Exception("none of the std ies can be < 0.4")
        pass

    for i in range(0, n):
        G.add_node(i)

        education, economic = generate_education_economic(mean=education_mean,
                                                          std=education_std,
                                                          edu_econ_correlation_val=edu_econ_correlation)

        xcor, ycor = position_news_consumer(economic, education)

        G.node[i] = {
            # "education": np.random.uniform(0, 3.01),
            # "economic": np.random.uniform(0, 3),
            # "colourList": "9830400,220",
            "color": "105",  # blue
            "driver": "-1.0",
            # if decided to envolve driver nodes make it 0 and uncommend the mds file in the main function
            "isProvider": "0.0",
            "education": str(round(education, 5)),
            "economic": str(round(economic, 5)),
            "WHO": str(float(i)),  # str(i),
            "XCOR": str(round(xcor, 5)),  # np.round(np.random.uniform(-1.0 * x_cor_max_min, x_cor_max_min), 3),
            "YCOR": str(round(ycor, 5)),  # np.round(np.random.uniform(-1.0 * y_cor_max_min, y_cor_max_min), 3),
        }

    edges = list(itertools.permutations(range(n), 2))
    random.shuffle(edges)
    edges_length = len(edges)

    # for edge in edges:
    while True:
        if (2.0 * G.number_of_edges()) / n >= target_k:
            break

        if edges_length == 0:
            break

        edge = edges.pop()
        edges_length -= 1
        n1 = edge[0]
        n2 = edge[1]

        if n1 == n2:
            continue

        education1 = float(G.node[n1]["education"])
        education2 = float(G.node[n2]["education"])
        economic1 = float(G.node[n1]["economic"])
        economic2 = float(G.node[n2]["economic"])

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
                "weight": str(weight),
                "label": str(weight),
                "color": "55"  # green
            })

    return G


def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
    pass


def stats(G):
    result = ""
    check_cutoff = 2.5
    edu = []
    for node in G.nodes():
        edu.append(float(G.node[node]["education"]))
        # if G.node[node]["education"] > 2:
        #     econ.add(node)
        #     pass

    edu_condition = []
    weights = []
    ppl_edu = []

    for node in G.nodes():
        ppl_edu.append(float(G.node[node]["education"]))
        pass

    count_nodes_with_no_link = len([val for val in G.degree().values() if val == 0])

    result += "$no-link-nodes-count = {} $count-ppl-edu > 2.5 = {} ".format(
        count_nodes_with_no_link,
        len([q for q in ppl_edu if q > 2.5]))

    edu_diffs = []
    econ_diffs = []
    for edge in G.edges():
        weights.append(float(G.edge[edge[0]][edge[1]]["weight"]))
        edu_diffs.append(np.absolute(float(G.node[edge[0]]["education"]) - float(G.node[edge[1]]["education"])))
        econ_diffs.append(np.absolute(float(G.node[edge[0]]["economic"]) - float(G.node[edge[1]]["economic"])))
        if float(G.node[edge[0]]["education"]) > check_cutoff and float(G.node[edge[1]]["education"]) > check_cutoff:
            edu_condition.append(np.absolute(float(G.node[edge[0]]["education"]) - float(G.node[edge[1]]["education"])))
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


def homophily_matrix_combined(G, save_p):
    def assign_group(val):
        if val > 0.67:
            return 2
        elif val < -0.67:
            return 0
        else:
            return 1
        pass

    def assign_combined_group(from_edu, to_edu, from_econ, to_econ):
        def combinations(edu_group, econ_group):
            i = -1
            if edu_group == 0 and econ_group == 0:
                i = 0
            elif edu_group == 0 and econ_group == 1:
                i = 1
            elif edu_group == 0 and econ_group == 2:
                i = 2
            elif edu_group == 1 and econ_group == 0:
                i = 3
            elif edu_group == 1 and econ_group == 1:
                i = 4
            elif edu_group == 1 and econ_group == 2:
                i = 5
            elif edu_group == 2 and econ_group == 0:
                i = 6
            elif edu_group == 2 and econ_group == 1:
                i = 7
            elif edu_group == 2 and econ_group == 2:
                i = 8

            return i
            pass

        i = combinations(
            edu_group=assign_group(from_edu),
            econ_group=assign_group(from_econ))
        j = combinations(
            edu_group=assign_group(to_edu),
            econ_group=assign_group(to_econ))

        return i, j
        pass

    nums = [[0 for x in range(9)] for y in range(9)]

    for edge in G.edges():
        i, j = assign_combined_group(
            from_edu=float(G.node[edge[0]]["education"]),
            to_edu=float(G.node[edge[1]]["education"]),
            from_econ=float(G.node[edge[0]]["economic"]),
            to_econ=float(G.node[edge[1]]["economic"])
        )

        nums[i][j] += 1

        pass

    df = DataFrame(nums)
    # df.rename(columns=lambda x: x * 0.1, inplace=True)
    # df.rename(dict(zip(df.index.tolist(), [round(x * 0.1, 1) for x in df.index.tolist()])), inplace=True)

    df.to_csv(save_p)

    pass


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
        edu_econ_correlation = float(sys.argv[17])
        pass

    if debug is True:
        n = 200
        target_k = 3.5
        # prob_cut_off = 0.940  #
        x_cor_max_min = 6.5  # control location belief
        y_cor_max_min = 6.5  # control location socio-econ belief etc
        links_weight_std = 0.09  # links weight between 0 0.2
        education_mean = 1.5
        education_std = 0.55
        economic_mean = 1.5
        economic_std = 0.55  # the smaller the less economic
        c = 3
        d = 1
        run_number = 0
        max_num_subscribers = 4
        subscription_weight_max_uniform = 0.3
        subscribers_inverse_distance_factor = 9.0
        edu_econ_correlation = 0.5
        pass

    G = using_permutation(
        n=n, target_k=target_k,
        c=c, d=d, x_cor_max_min=x_cor_max_min, y_cor_max_min=y_cor_max_min,
        links_weight_std=links_weight_std, education_mean=education_mean, education_std=education_std,
        economic_mean=economic_mean, economic_std=economic_std, edu_econ_correlation=edu_econ_correlation)

    # mds = GeneralTools.identify_driver_nodes(
    #     networkx_digraph=G, root_folder_work="pol", debug=False, draw_graphs=False,
    #     show_plots=False, network_id=1)

    homophily_matrix_combined(G, "d:\\temp\\homophily_matrix.csv")

    stats_val = stats(G)
    # tools.clipboard_copy(stats_val)
    if debug:
        print(stats_val)
        pass
    # n target_k x_cor_max_min y_cor_max_min links_weight_std education_mean education_std economic_mean economic_std c d debug run_number
    # 100 5 6 6 0.09 1.5 0.55 1.5 0.55 0.5 1.0 1 10

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
