import simpy
import numpy as np
from arena_server.channel_model import ChannelNetworkModel
from arena_server.radio import RadioEnvironment
from arena_server.simulation_logger import SimulationLogger
from typing import TYPE_CHECKING, Any, Optional, Generator, List
if TYPE_CHECKING:
    from arena_server.network_operator import NetworkOperator
    from arena_server.node import Node


class Simulator:
    def __init__(self, center_freq_list):
        self._center_freq = center_freq_list
        self._freq_channel_list = list(np.arange(len(self._center_freq)))
        self._des_env: simpy.Environment = simpy.Environment()
        self._simulation_logger = SimulationLogger(self)
        self._channel_network_model: ChannelNetworkModel = ChannelNetworkModel('Friis', self._center_freq)
        self._radio_env: 'RadioEnvironment' = RadioEnvironment(self, len(self._center_freq))
        self._node_list: List['Node'] = []
        self._network_operator_list: List['NetworkOperator'] = []

    @property
    def freq_channel_list(self) -> List:
        return self._freq_channel_list

    @property
    def logger(self) -> 'SimulationLogger':
        return self._simulation_logger

    @property
    def channel_network_model(self) -> 'ChannelNetworkModel':
        return self._channel_network_model

    @property
    def radio_env(self) -> 'RadioEnvironment':
        return self._radio_env

    def add_node(self, node: 'Node'):
        self._simulation_logger.logging({'type': 'add node', 'name': node.name, 'node type': node.node_type,
                                         'network operator': node.network_operator.name})
        node.set_simulator(self)
        self._node_list.append(node)
        self.channel_network_model.add_node_and_link(node)

    def add_network_operator(self, network_operator: 'NetworkOperator'):
        self._simulation_logger.logging({'type': 'add network operator', 'name': network_operator.name})
        network_operator.set_simulator(self)
        self._network_operator_list.append(network_operator)

    def register_process(self, generator: Generator[simpy.events.Event, Any, Any]):
        self._des_env.process(generator)

    def get_event(self) -> simpy.Event:
        return self._des_env.event()

    def get_timeout_event(self, delay: float, value: Optional[Any] = None) -> simpy.Timeout:
        return self._des_env.timeout(delay, value)

    def now(self) -> float:
        return self._des_env.now

    def run(self, until: float):
        self.logger.logging({'type': 'frequency', 'freq channel list': self._freq_channel_list})
        self.logger.logging({'type': 'start simulation'})
        self._des_env.run(until)


