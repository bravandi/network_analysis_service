import snap

def shortest_path_lenghts():
    Graph = snap.GenRndGnm(snap.PNGraph, 100, 1000)
    NIdToDistH = snap.TIntH()
    shortestPath = snap.GetShortPath(Graph, 10, NIdToDistH)
    for item in NIdToDistH:
        print item, NIdToDistH[item]

    print shortestPath


def plot_short_path_distribution():
    Graph = snap.GenRndGnm(snap.PNGraph, 100, 1000)
    snap.PlotShortPathDistr(Graph, "example", "Directed graph - shortest path")

    UGraph = snap.GenRndGnm(snap.PUNGraph, 100, 1000)
    snap.PlotShortPathDistr(UGraph, "example", "Undirected graph - shortest path")

    Network = snap.GenRndGnm(snap.PNEANet, 100, 1000)
    snap.PlotShortPathDistr(Network, "example", "Network - shortest path")


def DrawGViz():
    Graph = snap.GenRndGnm(snap.PNGraph, 10, 20)
    snap.DrawGViz(Graph, snap.gvlDot, "graph.png", "graph 1")

    UGraph = snap.GenRndGnm(snap.PUNGraph, 10, 40)
    snap.DrawGViz(UGraph, snap.gvlNeato, "graph_undirected.png", "graph 2", True)

    NIdColorH = snap.TIntStrH()
    NIdColorH[0] = "green"
    NIdColorH[1] = "red"
    NIdColorH[2] = "purple"
    NIdColorH[3] = "blue"
    NIdColorH[4] = "yellow"
    Network = snap.GenRndGnm(snap.PNEANet, 5, 10)
    snap.DrawGViz(Network, snap.gvlSfdp, "network.png", "graph 3", True, NIdColorH)

    # Graph = snap.GenRndGnm(snap.PNGraph, 10, 50)
    # labels = snap.TIntStrH()
    # for NI in Graph.Nodes():
    #     labels[NI.GetId()] = str(NI.GetId())
    # snap.DrawGViz(Graph, snap.gvlDot, "output.png", " ", labels)

    # UGraph = snap.GenRndGnm(snap.PUNGraph, 10, 50)
    # labels = snap.TIntStrH()
    # for NI in UGraph.Nodes():
    #     labels[NI.GetId()] = str(NI.GetId())
    # snap.DrawGViz(UGraph, snap.gvlDot, "output.png", " ", labels)
    #
    # Network = snap.GenRndGnm(snap.PNEANet, 10, 50)
    # labels = snap.TIntStrH()
    # for NI in Network.Nodes():
    #     labels[NI.GetId()] = str(NI.GetId())
    # snap.DrawGViz(Network, snap.gvlDot, "output.png", " ", labels)


if __name__ == '__main__':
    DrawGViz()