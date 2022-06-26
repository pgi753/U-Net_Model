from arena_server.controller import Controller
from arena_server.wifi_parameter import WiFiParam
from typing import TYPE_CHECKING, Dict, List
import numpy as np
if TYPE_CHECKING:
    from arena_server.wifi_network_operator import WiFiNetworkOperator


class WiFiCSMAController(Controller):
    def __init__(self, contention_window_size, num_unit_packet):
        super(WiFiCSMAController, self).__init__()
        self._contention_window_size = contention_window_size
        self._min_contention_window_size = contention_window_size
        self._max_contention_window_size = 65535
        self._num_unit_packet = num_unit_packet
        self._freq_channel_list = []
        self._primary_channel = 0
        self._back_off: int = np.random.randint(1, self._contention_window_size)
        self._cca_thresh = WiFiParam.CCA_THRESHOLD

    def set_network_operator(self, network_operator: 'WiFiNetworkOperator'):
        super(WiFiCSMAController, self).set_network_operator(network_operator)
        self._freq_channel_list = network_operator.freq_channel_list
        self._primary_channel = self._freq_channel_list[0]

    def request_decision(self, observation: Dict) -> Dict:
        action = {}
        observation_type = observation['type']
        if observation_type == 'sensing':
            sensed_power = observation['sensed_power']
            primary_sensed_power = sensed_power[self._primary_channel]
            if primary_sensed_power > self._cca_thresh:
                action = {'type': 'sensing'}
            else:
                self._back_off -= 1
                if self._back_off <= 0:
                    freq_channel_list = []
                    for ch in sensed_power:
                        if sensed_power[ch] <= self._cca_thresh:
                            freq_channel_list.append(ch)
                    action = {'type': 'tx_data_packet', 'freq_channel_list': freq_channel_list,
                              'num_unit_packet': self._num_unit_packet}
                else:
                    action = {'type': 'sensing'}
        elif observation_type == 'tx_data_packet':
            success_freq_channel_list = observation['success_freq_channel_list']
            if self._primary_channel in success_freq_channel_list:
                self._contention_window_size = self._min_contention_window_size
            else:
                self._contention_window_size = min(self._contention_window_size * 2,
                                                   self._max_contention_window_size)
            self._back_off = np.random.randint(1, self._contention_window_size)
            action = {'type': 'sensing'}
        return action


class WiFiPeriodicController(Controller):
    def __init__(self, num_back_off: int, num_unit_packet: int, frequency_channel_list: List[int]):
        super(WiFiPeriodicController, self).__init__()
        self._num_back_off = num_back_off
        self._num_unit_packet = num_unit_packet
        self._freq_channel_list = frequency_channel_list
        self._back_off = self._num_back_off

    def set_network_operator(self, network_operator: 'WiFiNetworkOperator'):
        super(WiFiPeriodicController, self).set_network_operator(network_operator)

    def request_decision(self, observation: Dict) -> Dict:
        action = {}
        observation_type = observation['type']
        if observation_type == 'sensing':
            self._back_off -= 1
            if self._back_off <= 0:
                action = {'type': 'tx_data_packet', 'freq_channel_list': self._freq_channel_list,
                          'num_unit_packet': self._num_unit_packet}
            else:
                action = {'type': 'sensing'}
        elif observation_type == 'tx_data_packet':
            self._back_off = self._num_back_off
            action = {'type': 'sensing'}
        return action

