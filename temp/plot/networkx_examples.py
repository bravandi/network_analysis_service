import networkx as nx
import complex_networks.tools as tools

if __name__ == '__main__':

    G = nx.complete_graph(4)

    tools.networkx_draw(G, 'ww.png')
