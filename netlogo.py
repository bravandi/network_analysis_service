import sys
from complex_networks.network import Network
from complex_networks.experiment import Experiment
import complex_networks.constants as constants


def identify_nodes_control_types(path, debug=False, draw_graphs=False):
    net = Network.networkx_create_from_gml(
        path=path
        # path="D:\\SoftwareProject\\complex_networks_tools\\data\\Neural Network\\celegansneural.gml"
    )

    ex = Experiment(debug=debug, draw_graphs=draw_graphs)

    cnetwork = ex.networkx_to_snap_cnetwork(
        networkx=net,
        name="",
        model=constants.NetworkModel.netlogo(),
        network_id=100
    )

    unmatched_nodes_inset, mds = cnetwork.control.snap_find_mds_minimum_driver_node_set()

    redundant_nodes, intermittent_nodes, critical_nodes = \
        cnetwork.control.snap_find_redundant_intermittent_critical_nodes()

    redundant_nodes_who = ''
    intermittent_nodes_who = ''
    critical_nodes_who = ''
    mds_who = ''

    for node in redundant_nodes:

        redundant_nodes_who += str(dict(net.nodes(True))[node]['WHO']) + ' '

    for node in intermittent_nodes:
        intermittent_nodes_who += str(dict(net.nodes(True))[node]['WHO']) + ' '

    for node in critical_nodes:
        critical_nodes_who += str(dict(net.nodes(True))[node]['WHO']) + ' '

    for node in mds:
        mds_who += str(dict(net.nodes(True))[node]['WHO']) + ' '

    result = "r:[%s]:r i:[%s]:i c:[%s]:c mds:[%s]:mds" % \
             (
                 redundant_nodes_who,
                 intermittent_nodes_who,
                 critical_nodes_who,
                 mds_who
             )

    print (result)


# node data
# dict(net.nodes(True))[101]
pass

if __name__ == '__main__':
    if len(sys.argv) != 2:
        identify_nodes_control_types('d:\\temp\\netlogo-diffusion.gml', debug=True, draw_graphs=True)
        # raise Exception("path must be given")
    else:
        identify_nodes_control_types(sys.argv[1], draw_graphs=True)
