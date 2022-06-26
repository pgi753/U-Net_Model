import numpy as np
from arena_server.node import Node
from arena_server.packet import UpperLayerPacket
from arena_server.wifi_parameter import WiFiParam
from typing import Optional, List


class WiFiApNode(Node):
    def __init__(self, name: str, freq_channel_list: List[int]):
        super(WiFiApNode, self).__init__(name, 'AP', 'tx_rx', WiFiParam.AP_ANTENNA_GAIN, WiFiParam.NOISE_POWER,
                                         freq_channel_list)
        self._state = 'sensing'
        self._sta_list: List['WiFiStaNode'] = []

    def add_station(self, sta: 'WiFiStaNode'):
        self._sta_list.append(sta)

    def _run(self):
        """
        sensing:
            Action parameter: None
            Operation: sense all frequency channels (duration: slot time)
            Observation: sensed power over all frequency channels (sensed_power)
        tx_data_packet:
            Action parameter: list of frequency channels (freq_channel_list), number of unit packets (num_unit_packet)
            Operation:
                1. Transmit data packets to randomly selected stations on a given list of frequency channels
                   (duration: UNIT_PACKET_DURATION * num_unit_packet)
                2. Wait for the acks from the stations (duration: SIFS + ACK_DURATION + SIFS)
            Observation: list of successful frequency channels (success_freq_channel_list)
        """
        event = None
        while True:
            observation = {}
            if self._state == 'sensing':
                observation = {'type': 'sensing', 'sensed_power': self.sensed_power}
            elif self._state == 'tx_data_packet':
                event = self.wait_for_time_out(WiFiParam.SIFS * 2 + WiFiParam.ACK_DURATION)
                yield event
                success_freq_channel_list = []
                packet = self.get_received_packet()
                while packet is not None:
                    upper_layer_packet = packet.upper_layer_packet
                    if upper_layer_packet.type == 'control' and upper_layer_packet.subtype == 'ack':
                        freq_channel = packet.freq_channel
                        success_freq_channel_list.append(freq_channel)
                    packet = self.get_received_packet()
                observation = {'type': 'tx_data_packet', 'success_freq_channel_list': success_freq_channel_list}
            action = self.request_decision_to_network_operator(observation)
            if action['type'] == 'sensing':
                event = self.start_multi_channel_sensing(WiFiParam.SLOT_TIME, self.freq_channel_list)
                self._state = 'sensing'
            elif action['type'] == 'tx_data_packet':
                freq_channel_list = action['freq_channel_list']
                # print(freq_channel_list)
                num_unit_packet = action['num_unit_packet']
                time_duration = WiFiParam.UNIT_PACKET_DURATION * num_unit_packet
                data_length = WiFiParam.DATA_BYTES_PER_UNIT_PACKET * num_unit_packet
                rx_node_list = []
                tx_power_list = []
                sinr_threshold_list = []
                upper_layer_packet_list = []
                for _ in freq_channel_list:
                    rx_node = np.random.choice(self._sta_list)
                    rx_node_list.append(rx_node)
                    tx_power = WiFiParam.AP_TX_POWER
                    tx_power_list.append(tx_power)
                    sinr_threshold_list.append(WiFiParam.SINR_THRESHOLD)
                    upper_layer_packet_list.append(UpperLayerPacket(type='data', data_length=data_length))
                event = self.send_multi_channel_phy_packet(rx_node_list, time_duration, freq_channel_list,
                                                           tx_power_list, sinr_threshold_list, upper_layer_packet_list)
                self._state = 'tx_data_packet'
            yield event


class WiFiStaNode(Node):
    def __init__(self, name: str, freq_channel_list: List[int]):
        super(WiFiStaNode, self).__init__(name, 'STA', 'tx_rx', WiFiParam.AP_ANTENNA_GAIN, WiFiParam.NOISE_POWER,
                                          freq_channel_list)
        self._ap: Optional[WiFiApNode] = None

    def set_ap(self, ap: 'WiFiApNode'):
        self._ap = ap

    def _run(self):
        while True:
            event = self.wait_for_packet_reception()
            yield event
            event = self.wait_for_time_out(WiFiParam.SIFS)
            yield event
            freq_channel_list = []
            time_duration = WiFiParam.ACK_DURATION
            rx_node_list = []
            tx_power_list = []
            sinr_threshold_list = []
            upper_layer_packet_list = []
            packet = self.get_received_packet()
            while packet is not None:
                upper_layer_packet = packet.upper_layer_packet
                if upper_layer_packet.type == 'data':
                    data_tx_node = packet.tx_node
                    freq_channel = packet.freq_channel
                    freq_channel_list.append(freq_channel)
                    ack_rx_node = data_tx_node
                    rx_node_list.append(ack_rx_node)
                    tx_power_list.append(WiFiParam.STA_TX_POWER)
                    sinr_threshold_list.append(WiFiParam.SINR_THRESHOLD)
                    upper_layer_packet_list.append(UpperLayerPacket(type='control', subtype='ack'))
                packet = self.get_received_packet()
            if freq_channel_list:
                event = self.send_multi_channel_phy_packet(rx_node_list, time_duration, freq_channel_list,
                                                           tx_power_list, sinr_threshold_list, upper_layer_packet_list)
            yield event
