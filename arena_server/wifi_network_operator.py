from typing import TYPE_CHECKING, Dict, List
from arena_server.wifi_node import WiFiApNode, WiFiStaNode
from arena_server.network_operator import NetworkOperator
if TYPE_CHECKING:
    from arena_server.simulator import Simulator


class WiFiNetworkOperator(NetworkOperator):
    def __init__(self, name: str, num_sta: int, freq_channel_list: List[int]):
        super(WiFiNetworkOperator, self).__init__(name)
        self._freq_channel_list = freq_channel_list
        # Generate AP and STAs
        ap_name = name + '_AP'
        self._ap = WiFiApNode(name=ap_name, freq_channel_list=freq_channel_list)
        self.add_node(self._ap)
        self._sta_list = []
        for ind in range(num_sta):
            sta_name = name + '_STA_' + str(ind)
            sta = WiFiStaNode(name=sta_name, freq_channel_list=freq_channel_list)
            self._ap.add_station(sta)
            sta.set_ap(self._ap)
            self._sta_list.append(sta)
            self.add_node(sta)

    @property
    def ap(self):
        return self._ap

    @property
    def sta_list(self):
        return self._sta_list

    @property
    def freq_channel_list(self):
        return self._freq_channel_list

    def get_information(self) -> Dict:
        info = {'sta list': [sta.name for sta in self._sta_list], 'freq channel list': self._freq_channel_list}
        return info
