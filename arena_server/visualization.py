import pyvista as pv
from pyvistaqt import BackgroundPlotter
import numpy as np
from typing import TYPE_CHECKING, Tuple, List, Dict, Optional
if TYPE_CHECKING:
    from arena_server.simulator import Simulator


class Node:
    def __init__(self, name: str, node_type: str, network_operator: 'NetworkOperator', plotter: BackgroundPlotter,
                 node_size: float):
        self._name = name
        self._node_type = node_type
        self._network_operator = network_operator
        self._position = np.array((0.0, 0.0))
        self._node_size = node_size
        if node_type == 'AP':
            self._mesh = pv.Cone(direction=[0, 0, 1], center=[0.0, 0.0, node_size],
                                 height=node_size * 8, radius=node_size * 2)
        else:
            self._mesh = pv.Sphere(radius=node_size, center=(0.0, 0.0, node_size))
        self._plotter = plotter
        self._actor = self._plotter.add_mesh(self._mesh, show_scalar_bar=False)

    @property
    def name(self):
        return self._name

    @property
    def network_operator(self):
        return self._network_operator

    @property
    def position(self):
        return np.array([self._position[0], self._position[1], self._node_size/2])

    def update_position(self, position: np.ndarray):
        prev_position = self._position
        self._position = position
        vec = position - prev_position
        self._mesh.translate([vec[0], vec[1], 0])


class NetworkOperator:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name


class LineObject:
    def __init__(self, plotter: BackgroundPlotter, color: str):
        self._active = False
        self._plotter = plotter
        self._mesh = pv.Line(np.array([0, 0, 0]), np.array([0, 0, 0]))
        self._plotter.add_mesh(self._mesh, color=color, show_scalar_bar=False)

    def show(self, start: np.ndarray, end: np.ndarray):
        self._active = True
        self._mesh.overwrite(pv.Line(start, end))

    def hide(self):
        self._active = False
        self._mesh.overwrite(pv.Line(np.array([0, 0, 0]), np.array([0, 0, 0])))


class Packet(LineObject):
    def __init__(self, plotter: BackgroundPlotter, color: 'str'):
        super(Packet, self).__init__(plotter, color)
        self._name = None
        self._time_duration = None
        self._freq_channel = None
        self._tx_node = None
        self._rx_node = None
        self._tx_power = None
        self._interference_list: List['Interference'] = []

    def set(self, name: str, time_duration: float, freq_channel: int, tx_node: 'Node', rx_node: 'Node',
            tx_power: float):
        self._name = name
        self._time_duration = time_duration
        self._freq_channel = freq_channel
        self._tx_node = tx_node
        self._rx_node = rx_node
        self._tx_power = tx_power
        self._interference_list = []
        self.show(tx_node.position, rx_node.position)

    @property
    def interference_list(self) -> List['Interference']:
        return self._interference_list

    def add_interference(self, interf: 'Interference'):
        self._interference_list.append(interf)

    def clear(self):
        self.hide()


class Interference(LineObject):
    def __init__(self, plotter: BackgroundPlotter, color: 'str'):
        super(Interference, self).__init__(plotter, color)
        self._name = None
        self._overlap_time = None
        self._freq_channel = None
        self._tx_node = None
        self._rx_node = None
        self._interference_energy = None

    def set(self, overlap_time: float, freq_channel: int, tx_node: 'Node', rx_node: 'Node',
            interference_energy: float):
        self._overlap_time = overlap_time
        self._freq_channel = freq_channel
        self._tx_node = tx_node
        self._rx_node = rx_node
        self._interference_energy = interference_energy
        self.show(tx_node.position, rx_node.position)

    def clear(self):
        self.hide()


class RadioNetwork:
    def __init__(self, max_num_packet: int, max_num_interference: int, plotter: BackgroundPlotter):
        self._packet_pool: List['Packet'] = [Packet(plotter, 'white') for _ in range(max_num_packet)]
        self._interference_pool: List['Interference'] = [Interference(plotter, 'red')
                                                         for _ in range(max_num_interference)]
        self._active_packet_dict: Dict[str, 'Packet'] = {}
        self._active_interference_list: List['Interference'] = []

    def add_packet(self, time_duration: float, freq_channel: int, tx_node: 'Node', rx_node: 'Node', tx_power: float):
        name = tx_node.name + '_' + rx_node.name + '_' + str(freq_channel)
        if self._packet_pool and (name not in self._active_packet_dict):
            packet = self._packet_pool.pop()
            packet.set(name, time_duration, freq_channel, tx_node, rx_node, tx_power)
            self._active_packet_dict[name] = packet

    def add_interference(self, overlap_time: float, freq_channel: int, tx_packet_nodes: Tuple['Node', 'Node'],
                         rx_packet_nodes: Tuple['Node', 'Node'], interference_energy: float):
        tx_packet_name = tx_packet_nodes[0].name + '_' + tx_packet_nodes[1].name + '_' + str(freq_channel)
        rx_packet_name = rx_packet_nodes[0].name + '_' + rx_packet_nodes[1].name + '_' + str(freq_channel)
        tx_packet = self._active_packet_dict.get(tx_packet_name, None)
        rx_packet = self._active_packet_dict.get(rx_packet_name, None)
        if (tx_packet is not None) and (rx_packet is not None) and self._interference_pool:
            interf = self._interference_pool.pop()
            interf.set(overlap_time, freq_channel, tx_packet_nodes[0], rx_packet_nodes[1], interference_energy)
            self._active_interference_list.append(interf)
            tx_packet.add_interference(interf)
            rx_packet.add_interference(interf)

    def remove_packet(self, freq_channel: int, tx_node: 'Node', rx_node: 'Node'):
        name = tx_node.name + '_' + rx_node.name + '_' + str(freq_channel)
        if name in self._active_packet_dict:
            packet = self._active_packet_dict.pop(name)
            for interf in packet.interference_list:
                if interf in self._active_interference_list:
                    interf.clear()
                    self._active_interference_list.remove(interf)
                    self._interference_pool.append(interf)
            packet.clear()
            self._packet_pool.append(packet)


class RadioTime:
    def __init__(self, plotter: BackgroundPlotter, num_time_step: int = 1000, cell_time_size: float = 1,
                 cell_freq_size: float = 30, cell_height: float = 50, freq_separation: float = 20):
        self._current_occupancy: Dict[str, np.ndarray] = {}
        self._freq_channel_list: Optional[List[int]] = None
        self._network_operator_list: Optional[List[str]] = None
        self._time_domain_occupancy: Optional[Dict[str, np.ndarray]] = None
        self._num_time_step = num_time_step
        self._grid_x: Dict[str, np.ndarray] = {}
        self._grid_y: Dict[str, np.ndarray] = {}
        cell_margin: float = 0.01
        self._cell_unit = cell_height * np.array([[0, 1, 1, 0], [0, 1, 1, 0]])
        self._grid_unit_x = np.array([[0, cell_margin, cell_margin + cell_freq_size, 2 * cell_margin + cell_freq_size],
                                      [0, cell_margin, cell_margin + cell_freq_size, 2 * cell_margin + cell_freq_size]])
        self._grid_unit_y = np.array([[0, 0, 0, 0], [cell_time_size, cell_time_size, cell_time_size, cell_time_size]])
        self._grid_spacing_x = 2 * cell_margin + cell_freq_size + freq_separation
        self._grid_spacing_y = cell_margin + cell_time_size
        self._mesh_dict = {}
        self._plotter = plotter

    def set_freq_channel_list(self, freq_channel_list: List[int]):
        self._freq_channel_list = freq_channel_list

    def set_network_operator_list(self, network_operator_list: List[str]):
        self._network_operator_list = network_operator_list
        freq_channel_num = len(self._freq_channel_list)
        occupancy = np.zeros((2, 4 * freq_channel_num))
        self._current_occupancy = {network_operator: occupancy.copy() for network_operator in
                                   self._network_operator_list}
        time_occupancy = np.zeros((2 * self._num_time_step, 4 * freq_channel_num))
        self._time_domain_occupancy = {network_operator: time_occupancy.copy() for network_operator in
                                       self._network_operator_list}
        max_grid_x = 0
        max_grid_y = 0
        for ind, network_operator in enumerate(self._network_operator_list):
            self._grid_x[network_operator] = np.zeros_like(time_occupancy)
            self._grid_y[network_operator] = np.zeros_like(time_occupancy)
            for time in range(self._num_time_step):
                for freq in range(freq_channel_num):
                    self._grid_x[network_operator][2*time: 2*time+2, 4*freq: 4*freq+4] = \
                        self._grid_unit_x + self._grid_spacing_x * freq
                    self._grid_y[network_operator][2*time: 2*time+2, 4*freq: 4*freq+4] = \
                        self._grid_unit_y + self._grid_spacing_y * time
            max_grid_x = np.max((max_grid_x, np.max(self._grid_x[network_operator])))
            max_grid_y = np.max((max_grid_y, np.max(self._grid_y[network_operator])))

        self._plotter.subplot(0, 1)
        self._plotter.add_mesh(pv.Box([0, max_grid_x, 0, max_grid_y, -0.01, 0.01]), color='black')
        color = ['red', 'blue', 'yellow', 'green', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan']
        for ind, network_operator in enumerate(self._network_operator_list):
            mesh = pv.StructuredGrid(self._grid_x[network_operator], self._grid_y[network_operator],
                                     self._time_domain_occupancy[network_operator])
            self._mesh_dict[network_operator] = mesh
            self._plotter.add_mesh(mesh, color=color[ind], opacity=1)

    def send_packet(self, network_operator: str, freq_channel: int):
        ind = self._freq_channel_list.index(freq_channel)
        self._current_occupancy[network_operator][0: 2, 4*ind: 4*(ind+1)] = self._cell_unit

    def receive_packet(self, network_operator: str, freq_channel: int):
        ind = self._freq_channel_list.index(freq_channel)
        self._current_occupancy[network_operator][0: 2, 4*ind: 4*(ind+1)] = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

    def update(self):
        for network_operator in self._network_operator_list:
            time_occupancy = self._time_domain_occupancy[network_operator]
            occupancy = self._current_occupancy[network_operator]
            time_occupancy[:] = np.row_stack((time_occupancy[2:, :], occupancy))
            points = np.column_stack((self._grid_x[network_operator].flatten(order='F'),
                                      self._grid_y[network_operator].flatten(order='F'),
                                      time_occupancy.flatten(order='F')))
            self._mesh_dict[network_operator].points = points


class Visualization:
    def __init__(self, refresh_rate: float, max_num_line_object: int, plane_size: Tuple[float, float], node_size: float,
                 activate_time_domain_view: bool = False):
        self._freq_channel_list: List = []
        self._network_operator_dict = {}
        self._node_dict: Dict[str, Node] = {}
        self._now = 0.0
        self._node_size = node_size
        self._radio_time = None
        if activate_time_domain_view:
            self._plotter = BackgroundPlotter(auto_update=refresh_rate, shape=(1, 2))
            self._radio_time = RadioTime(self._plotter)
        else:
            self._plotter = BackgroundPlotter(auto_update=refresh_rate)
        self._radio_network = RadioNetwork(max_num_line_object, max_num_line_object, self._plotter)
        self._plane = pv.Plane(i_size=plane_size[0], j_size=plane_size[1], i_resolution=1, j_resolution=1)
        self._plotter.add_mesh(self._plane, show_scalar_bar=False)
        view_size = max(plane_size)
        self._plotter.set_position((0, -1.5 * view_size, 1.5 * view_size))
        self._plotter.set_focus((0, 0, -0.1 * view_size))
        self._time_text = self._plotter.add_text('')
        self._time_text.SetMaximumFontSize(20)

    def __call__(self, log):
        log_type = log['type']
        self._now = log['time']
        now_sec = int(self._now / 1000000)
        now_msec = int(self._now / 1000)
        now_usec = int(np.round(self._now) % 1000)
        self._time_text.SetText(2, f'{now_sec: 04d} s\n{now_msec: 04d} ms\n{now_usec: 04d} us')
        if log_type == 'frequency':
            self._set_freq_channel_list(log)
        elif log_type == 'add network operator':
            self._add_network_operator(log)
        elif log_type == 'add node':
            self._add_node(log)
        elif log_type == 'movement':
            self._movement(log)
        elif log_type == 'start simulation':
            self._start_simulation()
        elif log_type == 'send packet':
            self._send_packet(log)
        elif log_type == 'receive packet':
            self._receive_packet(log)
        elif log_type == 'interference':
            self._interference(log)
        elif log_type == 'clock':
            self._clock()

    def _set_freq_channel_list(self, log):
        freq_channel_list = log['freq channel list']
        self._freq_channel_list = freq_channel_list
        if self._radio_time is not None:
            self._radio_time.set_freq_channel_list(freq_channel_list)

    def _add_network_operator(self, log):
        name = log['name']
        network_operator = NetworkOperator(name)
        self._network_operator_dict[name] = network_operator

    def _add_node(self, log):
        name = log['name']
        node_type = log['node type']
        network_operator = self._network_operator_dict[log['network operator']]
        node = Node(name, node_type, network_operator, self._plotter, self._node_size)
        self._node_dict[name] = node

    def _movement(self, log):
        node = self._node_dict[log['node']]
        position = np.array(log['position'])
        node.update_position(position)

    def _start_simulation(self):
        if self._radio_time is not None:
            self._radio_time.set_network_operator_list(list(self._network_operator_dict))

    def _send_packet(self, log):
        time_duration = log['time duration']
        freq_channel = log['freq channel']
        tx_node = self._node_dict[log['tx node']]
        rx_node = self._node_dict[log['rx node']]
        tx_power = log['tx power']
        network_operator_name = tx_node.network_operator.name
        self._radio_network.add_packet(time_duration, freq_channel, tx_node, rx_node, tx_power)
        if self._radio_time is not None:
            self._radio_time.send_packet(network_operator_name, freq_channel)

    def _interference(self, log):
        overlap_time = log['overlap time']
        freq_channel = log['freq channel']
        tx_packet_name = log['tx packet']
        rx_packet_name = log['rx packet']
        tx_packet_nodes = (self._node_dict[tx_packet_name[0]], self._node_dict[tx_packet_name[1]])
        rx_packet_nodes = (self._node_dict[rx_packet_name[0]], self._node_dict[rx_packet_name[1]])
        interference_energy = log['interference energy']
        self._radio_network.add_interference(overlap_time, freq_channel, tx_packet_nodes, rx_packet_nodes, interference_energy)

    def _receive_packet(self, log):
        freq_channel = log['freq channel']
        tx_node = self._node_dict[log['tx node']]
        rx_node = self._node_dict[log['rx node']]
        network_operator_name = tx_node.network_operator.name
        self._radio_network.remove_packet(freq_channel, tx_node, rx_node)
        if self._radio_time is not None:
            self._radio_time.receive_packet(network_operator_name, freq_channel)

    def _clock(self):
        if self._radio_time is not None:
            self._radio_time.update()
