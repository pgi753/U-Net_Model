import numpy as np
import simpy
from typing import TYPE_CHECKING, Optional, Dict, List, Union
from arena_server.packet import PhyPacket, UpperLayerPacket
if TYPE_CHECKING:
    from arena_server.simulator import Simulator
    from arena_server.network_operator import NetworkOperator
    from arena_server.data_queue import DataQueue


class Node:
    """ Node class
    """
    def __init__(self, name: str, node_type: str, trx_type: str, antenna_gain: float, noise_power: float,
                 frequency_channel_list: List[int]):
        """
        Args:
            name: name of the node
            node_type: type of the node (e.g., AP and STA)
            trx_type: channel gain is not calculated between (tx and tx) or (rx and rx).
                      'tx'-> transmit only, 'rx'-> receive only, 'tx_rx'-> transmit and receive
            antenna_gain: antenna gain (dB)
            noise_power: noise power (mW)
            frequency_channel_list: list of available frequency channels
        """
        self._name = name
        self._node_type = node_type
        self._trx_type = trx_type
        self._antenna_gain = antenna_gain
        self._noise_power = noise_power
        self._freq_channel_list: List[int] = frequency_channel_list
        self._simulator: Optional['Simulator'] = None
        self._network_operator: Optional['NetworkOperator'] = None
        self._position = np.array((0.0, 0.0))
        self._data_queue_list: List[DataQueue] = []
        self._sensed_power: Optional[Dict[int, float]] = {ch: 0 for ch in self._freq_channel_list}
        self._packet_waiting_event: Optional[simpy.Event] = None
        self._received_packet_list: List[PhyPacket] = []

    def set_simulator(self, simulator: 'Simulator'):
        self._simulator = simulator
        self._simulator.register_process(self._run())
        for queue in self._data_queue_list:
            queue.set_simulator(simulator)

    def set_network_operator(self, network_operator: 'NetworkOperator'):
        self._network_operator = network_operator

    @property
    def name(self) -> str:
        return self._name

    @property
    def node_type(self) -> str:
        return self._node_type

    @property
    def trx_type(self) -> str:
        return self._trx_type

    @property
    def freq_channel_list(self) -> List[int]:
        return self._freq_channel_list

    @property
    def network_operator(self) -> 'NetworkOperator':
        return self._network_operator

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, position):
        self._position = np.array(position)

    @property
    def sensed_power(self) -> Dict[int, float]:
        return self._sensed_power

    @property
    def simulator(self) -> 'Simulator':
        return self._simulator

    def get_antenna_gain(self) -> float:
        return self._antenna_gain

    def add_data_queue(self, data_queue: 'DataQueue'):
        data_queue.set_node(self)
        self._data_queue_list.append(data_queue)

    def send_multi_channel_phy_packet(self, rx_node_list: List[Union['Node', List['Node']]],
                                      time_duration: float, freq_channel_list: List[int],
                                      tx_power_list: List[float], sinr_threshold_list: List[float],
                                      upper_layer_packet_list: List[UpperLayerPacket]) -> simpy.Timeout:
        """ Send physical layer packet to one or multiple target receive node on multiple frequency channels
        Args:
            rx_node_list: List of receive node(s) of the packet on multiple frequency channels
            time_duration: time duration of packet
            freq_channel_list: List of frequency channel of packet
            tx_power_list: List of transmit power of packet
            sinr_threshold_list: List of sinr threshold of packet
            upper_layer_packet_list: List of upper layer packet enclosed in the physical layer packet
        Returns:
            timeout event that is triggered when packet transmission is over
        """
        time_out_event = None
        for ind, freq_channel in enumerate(freq_channel_list):
            time_out_event = self.send_phy_packet(rx_node_list[ind], time_duration, freq_channel,
                                                  tx_power_list[ind], sinr_threshold_list[ind],
                                                  upper_layer_packet_list[ind])
        return time_out_event

    def send_phy_packet(self, rx_node: Union['Node', List['Node']], time_duration: float, freq_channel: int,
                        tx_power: float, sinr_threshold: float, upper_layer_packet: UpperLayerPacket) -> simpy.Timeout:
        """ Send physical layer packet to one or multiple target receive node
        Args:
            rx_node: receive node of the packet. it is a list of nodes if it is sent to multiple receive nodes.
            time_duration: time duration of packet
            freq_channel: frequency channel of packet
            tx_power: transmit power of packet
            sinr_threshold: sinr threshold of packet
            upper_layer_packet: upper layer packet enclosed in the physical layer packet
        Returns:
            timeout event that is triggered when packet transmission is over
        """
        upper_layer_type = upper_layer_packet.type
        upper_layer_subtype = upper_layer_packet.subtype
        if upper_layer_subtype is not None:
            upper_layer_type = upper_layer_type + '.' + upper_layer_subtype
        rx_node_list = rx_node if isinstance(rx_node, list) else [rx_node]
        for rn in rx_node_list:
            self._simulator.logger.logging({'type': 'send packet', 'tx node': self.name, 'rx node': rn.name,
                                            'time duration': time_duration, 'freq channel': freq_channel,
                                            'tx power': tx_power, 'upper_layer_type': upper_layer_type})
        packet = PhyPacket(self, rx_node, time_duration, freq_channel, tx_power, sinr_threshold, upper_layer_packet)
        return self._simulator.radio_env.send_packet(packet)

    def receive_phy_packet(self, packet: PhyPacket, rx_power: float, interference: float):
        """ Receive packet
        Called from RadioEnvironment when the radio signal transmission containing a packet is over.
        Receive packet along with receive power and interference.
        Calculate SINR based on receive power and interference and decide if SINR is beyond the threshold.
        If the packet is correctly received, it is queued in the receive packet list, and the node is interrupted
        in the case that it is waiting for the packet reception.
        Args:
            packet: received packet
            rx_power: receive power of the packet
            interference: interference power
        """
        sinr = 10 * np.log10(rx_power / (interference + self._noise_power))
        if sinr > packet.sinr_threshold:
            self._received_packet_list.insert(0, packet)
            self.packet_reception_interrupt()
            result = 'success'
        else:
            result = 'failure'
        upper_layer_type = packet.upper_layer_packet.type
        upper_layer_subtype = packet.upper_layer_packet.subtype
        if upper_layer_subtype is not None:
            upper_layer_type = upper_layer_type + '.' + upper_layer_subtype
        self._simulator.logger.logging({'type': 'receive packet', 'result': result, 'tx node': packet.tx_node.name,
                                        'rx node': self.name, 'time duration': packet.time_duration,
                                        'freq channel': packet.freq_channel, 'sinr threshold': packet.sinr_threshold,
                                        'sinr': sinr, 'rx power': rx_power, 'interference': interference,
                                        'upper layer type': upper_layer_type})

    def get_received_packet(self) -> Optional[PhyPacket]:
        if len(self._received_packet_list) > 0:
            return self._received_packet_list.pop()
        else:
            return None

    def start_multi_channel_sensing(self, time_duration: float, freq_channel_list: List[int]) -> simpy.Timeout:
        """ Start channel sensing on multiple frequency channels
        Args:
            time_duration: time duration of sensing
            freq_channel_list: list of frequency channels for sensing
        Returns:
            timeout event that is triggered when sensing is over
        """
        time_out_event = None
        for freq_channel in freq_channel_list:
            time_out_event = self.start_sensing(time_duration, freq_channel)
        return time_out_event

    def start_sensing(self, time_duration: float, freq_channel: int) -> simpy.Timeout:
        """ Start channel sensing on a single frequency channel
        Args:
            time_duration: time duration of sensing
            freq_channel: frequency channel for sensing
        Returns:
            timeout event that is triggered when sensing is over
        """
        self._simulator.logger.logging({'type': 'start sensing', 'node': self.name, 'time duration': time_duration,
                                        'freq channel': freq_channel})
        return self._simulator.radio_env.start_sensing(self, time_duration, freq_channel)

    def finish_sensing(self, sensed_power: float, time_duration: float, freq_channel: int):
        """ Receive packet
        Called from RadioEnvironment when the sensing is over.
        Args:
            sensed_power: average sensed power during sensing time period
            time_duration: time duration of sensing
            freq_channel: frequency channel of sensed power
        """
        self._simulator.logger.logging({'type': 'finish sensing', 'node': self.name, 'time duration': time_duration,
                                        'freq channel': freq_channel, 'sensed power': sensed_power})
        self._sensed_power[freq_channel] = sensed_power

    def wait_for_time_out(self, wait_time: float) -> simpy.Timeout:
        """ Wait for given time
        Args:
             wait_time: time to wait
        Returns:
            timeout event that is triggered after wait_time
        """
        return self._simulator.get_timeout_event(wait_time)

    def wait_for_packet_reception(self) -> simpy.Event:
        """ Waiting until a packet is received
        Returns:
            event that is triggered when a packet is received
        """
        self._packet_waiting_event = self._simulator.get_event()
        return self._packet_waiting_event

    def packet_reception_interrupt(self):
        """ Node is interrupted when it is waiting for the packet reception
        """
        if self._packet_waiting_event is not None:
            self._packet_waiting_event.succeed()
            self._packet_waiting_event = None

    def request_decision_to_network_operator(self, observation: Dict) -> Dict:
        action = self._network_operator.request_decision(self, observation)
        return action

    def request_decision_to_network_operator_2(self, observation: Dict) -> Dict:
        action = self._network_operator.request_decision(self, observation)
        return action

    def _run(self):
        raise NotImplementedError
