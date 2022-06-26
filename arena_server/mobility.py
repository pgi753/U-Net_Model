import numpy as np
from typing import TYPE_CHECKING, Tuple, List, Optional

if TYPE_CHECKING:
    from arena_server.simulator import Simulator
    from arena_server.node import Node
    from arena_server.network_operator import NetworkOperator
    from arena_server.wifi_network_operator import WiFiNetworkOperator


class Mobility:
    """Mobility base class
    """
    def __init__(self):
        self._simulator: Optional[Simulator] = None
        self._network_operator: Optional[NetworkOperator] = None

    def set_simulator(self, simulator: 'Simulator'):
        """ Set simulator to the mobility object
        Register mobility process to DES
        Args:
            simulator: Simulator object
        """
        self._simulator = simulator
        self._initialize()
        simulator.register_process(self._run())

    def set_network_operator(self, network_operator: 'NetworkOperator'):
        """ Set network operator to the mobility object
        Args:
            network_operator: Network operator object
        """
        self._network_operator = network_operator
        self._bring_nodes_from_network_operator(network_operator)

    def _bring_nodes_from_network_operator(self, network_operator: 'NetworkOperator'):
        """ Bring nodes form the network operator
        Args:
            network_operator: Network operator object
        """
        raise NotImplementedError

    def _report_and_log_moved_nodes(self, moved_node_list: List['Node']):
        """ Report moved nodes to channel_network_model for marking as channel update needed.
        Log the moved nodes and their new positions.
        Args:
            moved_node_list: List of all moved nodes
        """
        self._simulator.channel_network_model.mark_update_needed(moved_node_list)
        # Logging movement of stations
        for node in moved_node_list:
            self._simulator.logger.logging({'type': 'movement', 'node': node.name, 'position': node.position.tolist()})

    def _initialize(self):
        """ Initialize the positions of all nodes
        """
        raise NotImplementedError

    def _run(self):
        """ Generator for the process in DES
        """
        raise NotImplementedError


class WiFiRandomWaypointMobility(Mobility):
    """ Random waypoint mobility model for WiFi
    Area is defined as a disk around the AP. Stations randomly move according to the random waypoint model.
    """
    def __init__(self, area_center: Tuple[float, float], area_radius: float, speed_range: Tuple[float, float],
                 avg_pause_time: float, update_interval: float):
        """
        Args:
            area_center: Area center position (meter)
            area_radius: Area radius (meter)
            speed_range: Tuple of (lowest speed, highest speed) of stations (km/h)
            avg_pause_time: Average pause time (ms)
            update_interval: Position update interval (ms)
        """
        super(WiFiRandomWaypointMobility, self).__init__()
        self._area_center = np.array(area_center)
        self._area_radius = area_radius
        self._speed_range = speed_range
        self._avg_pause_time = avg_pause_time
        self._update_interval = update_interval
        self._ap: Optional['Node'] = None
        self._sta_list: List['Node'] = []
        self._num_sta = 0
        self._sta_movement_vector = np.ndarray((0,))
        self._sta_movement_count = np.ndarray((0,))
        self._sta_pause_count = np.ndarray((0,))

    def _bring_nodes_from_network_operator(self, network_operator: 'WiFiNetworkOperator'):
        """ Bring AP and stations objects from the network operator and initialize the positions of them
        Args:
            network_operator: Wifi network operator object
        """
        self._ap = network_operator.ap
        self._sta_list = network_operator.sta_list
        self._num_sta = len(self._sta_list)
        self._sta_movement_vector = np.zeros((self._num_sta, 2))
        self._sta_movement_count = np.zeros(self._num_sta)
        self._sta_pause_count = np.zeros(self._num_sta)

    def _initialize(self):
        """ Initialize the positions of all nodes to the random positions within a disk
        """
        self._ap.position = self._area_center
        sta_position = self._random_position(self._num_sta)
        for ind, sta in enumerate(self._sta_list):
            sta.position = sta_position[ind, :]
        self._report_and_log_moved_nodes([self._ap] + self._sta_list)

    def _run(self):
        """ Generator for the process in DES
        Each station moves by sta_movement_vector at each step until sta_movement_count counts down to zero.
        Then, it waits for sta_pause_count steps.
        When sta_movement_count and sta_pause_count become zero, new position and speed are randomly selected and
        sta_movement_vector and sta_movement_count are calculated accordingly,
        and sta_pause_count is randomly selected as well.
        Moved stations are marked in the channel network model so that the channels for these stations are recalculated.
        """
        # if the update interval is equal to or less than zero, there is no location update.
        if self._update_interval <= 0:
            yield self._simulator.get_event()
        while True:
            moved_sta_list = []
            for ind, sta in enumerate(self._sta_list):
                if self._sta_movement_count[ind] == 0 and self._sta_pause_count[ind] == 0:
                    speed = np.random.uniform(self._speed_range[0], self._speed_range[1])  # speed in km/h
                    new_position = self._random_position(1)
                    vec = new_position - sta.position
                    dist = np.linalg.norm(vec)
                    self._sta_movement_count[ind] = np.ceil((dist / speed * 3600) / self._update_interval)
                    self._sta_movement_vector[ind, :] = vec / self._sta_movement_count[ind]
                    self._sta_pause_count[ind] = np.ceil(np.random.exponential(self._avg_pause_time)
                                                        / self._update_interval)
                if self._sta_movement_count[ind] > 0:
                    self._sta_list[ind].position += self._sta_movement_vector[ind, :]
                    self._sta_movement_count[ind] -= 1
                    moved_sta_list.append(self._sta_list[ind])
                elif self._sta_pause_count[ind] > 0:
                    self._sta_pause_count[ind] -= 1
            self._report_and_log_moved_nodes(moved_sta_list)
            event = self._simulator.get_timeout_event(self._update_interval)
            yield event

    def _random_position(self, num: int) -> np.array:
        """ Generate random positions in a disk
        Args:
            num: number of random positions
        Returns:
            random positions
        """
        radius = np.random.random(num) * self._area_radius
        angle = np.random.random(num) * 2 * np.pi
        position_x = radius * np.cos(angle) + self._area_center[0]
        position_y = radius * np.sin(angle) + self._area_center[1]
        return np.stack((position_x, position_y), axis=1)
