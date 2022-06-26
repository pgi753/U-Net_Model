from typing import TYPE_CHECKING, List, Optional, Dict
from arena_server.node import Node
from arena_server.controller import Controller

if TYPE_CHECKING:
    from arena_server.simulator import Simulator
    from arena_server.mobility import Mobility


class NetworkOperator:
    def __init__(self, name: str):
        self._name: str = name
        self._simulator: Optional['Simulator'] = None
        self._node_list: List['Node'] = []
        self._mobility: Optional['Mobility'] = None
        self._controller: Optional['Controller'] = None

    @property
    def name(self):
        return self._name

    @property
    def simulator(self):
        return self._simulator

    @property
    def controller(self):
        return self._controller

    def set_simulator(self, simulator: 'Simulator'):
        self._simulator = simulator
        for node in self._node_list:
            simulator.add_node(node)
        if self._mobility is not None:
            self._mobility.set_simulator(simulator)

    def set_mobility(self, mobility: 'Mobility'):
        self._mobility = mobility
        mobility.set_network_operator(self)
        if self._simulator is not None:
            self._mobility.set_simulator(self._simulator)

    def set_controller(self, controller: 'Controller'):
        self._controller = controller
        controller.set_network_operator(self)

    def add_node(self, node: 'Node'):
        node.set_network_operator(self)
        self._node_list.append(node)

    def get_information(self) -> Dict:
        raise NotImplementedError

    def request_decision(self, node: 'Node', observation: Dict) -> Dict:
        return self.controller.request_decision(observation)
