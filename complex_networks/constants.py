path_work = 'D:/SoftwareProject/network_analysis_service/'
path_bipartite_representations = path_work + 'temp/work/bipartite_representations/'
path_bipartite_matching_app_snap = path_work + 'temp/work/snap_bipartite_matching_flows.exe'
max_flow_sink_id = 63472995 #63,472,995
max_flow_source_id = max_flow_sink_id - 1
# 999999999  SNAP push relabel algorithm rejects this number as a node id
# 99999999


class NetworkModel:
    @staticmethod
    def erdos_renyi():
        return "erdos_renyi"

    @staticmethod
    def scale_free():
        return "scale_free"

    @staticmethod
    def real_network():

        return "real_network"

    @staticmethod
    def netlogo():

        return "netlogo"

    @staticmethod
    def bipartite_matching():

        return "bipartite_matching"

    @staticmethod
    def parse(s):
        s = s.lower()

        if s in ['netlogo']:
            return NetworkModel.netlogo()

        if s in ['bipartite_matching']:
            return NetworkModel.bipartite_matching()

        if s in ["erdos_renyi", 'er', 'erdos-renyi']:
            return NetworkModel.erdos_renyi()

        if s in ["scale_free", 'er', 'scale-free']:
            return NetworkModel.scale_free()

        if s in ["real_network", "rn"]:
            return NetworkModel.real_network()

        return None