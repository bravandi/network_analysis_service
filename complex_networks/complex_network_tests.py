from complex_networks.complex_network import ComplexNetwork
import complex_networks.tools as tools
import json

import common.event as event
import random
# from complex_networks.complex_network import NetworkModel, ComplexNetwork
import numpy
import numpy as np


if __name__ == '__main__':

    complex_net = ComplexNetwork.create_from_text_file(
        path='D:\\SoftwareProject\\complex_networks_tools\\data\\High-energy physics theory citation network\\Cit-HepTh.txt',
        experiment=None,
        name='High-energy physics theory citation network',
        from_to_separator='\t'
    )

    driver_nodes = complex_net.driver_nodes_maximal_matching()

    with open('d:\\data.txt', 'w') as outfile:
        json.dump(driver_nodes, outfile)
