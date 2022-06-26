import numpy as np
import networkx as nx
from typing import TYPE_CHECKING, Tuple, List, Union
if TYPE_CHECKING:
    from arena_server.node import Node


class ChannelModelFactory:
    """Factory class for channel model
    """
    def __init__(self, channel_model_type: str, center_freq: Union[np.ndarray, List]):
        """
        Args:
            channel_model_type: type of channel model (e.g., Friis)
            center_freq: 1D array of center frequencies of all frequency channels (MHz)
        """
        self._channel_model_type = channel_model_type
        self._center_freq = center_freq

    def get_channel_model(self, link: Tuple['Node', 'Node']):
        """
        Args:
            link: tuple of two nodes which the channel connects
        Returns:
            channel model
        """
        if self._channel_model_type == 'Friis':
            channel_model = FriisChannelModel(self._center_freq, link)
        else:
            channel_model = None
        return channel_model


class ChannelModel:
    """Base channel model
    """
    def __init__(self, center_freq: Union[np.ndarray, List], link: Tuple['Node', 'Node']):
        """
        Args:
            center_freq: 1D array of center frequencies of all frequency channels (MHz)
            link: tuple of two nodes which this channel connects
        """
        self._center_freq = np.array(center_freq)
        self._link = link
        self._num_freq_channel = len(self._center_freq)
        c = 299792458  # speed of light (m/s)
        self._wave_len = c / self._center_freq * (10 ** (-6))  # wave length (meter)
        self._channel_gain = np.zeros(self._num_freq_channel)
        self._update_needed = [True] * self._num_freq_channel

    def get_channel_gain(self, freq_channel: int) -> float:
        """Get channel gain for the given frequency channel.
        If the channel gain needs update, it would be updated and returned.
        Args:
            freq_channel: frequency channel index
        Returns:
            channel gain
        """
        if self._update_needed[freq_channel]:
            self._update_channel_gain(freq_channel)
            self._update_needed[freq_channel] = False
        return self._channel_gain[freq_channel]

    def _update_channel_gain(self, freq_channel: int):
        """Update channel gain for the given frequency channel.
        Args:
            freq_channel: frequency channel index
        """
        raise NotImplementedError

    def mark_update_needed(self):
        """Mark that the update is needed.
        """
        self._update_needed = [True] * self._num_freq_channel


class FriisChannelModel(ChannelModel):
    """Friis channel model
    """
    def __init__(self, center_freq: Union[np.ndarray, List], link: Tuple['Node', 'Node']):
        super(FriisChannelModel, self).__init__(center_freq, link)

    def _update_channel_gain(self, freq_channel: int):
        dist = np.linalg.norm(self._link[0].position - self._link[1].position)
        antenna_gain = 10.0 ** ((self._link[0].get_antenna_gain() + self._link[1].get_antenna_gain()) / 10.0)
        self._channel_gain[freq_channel] = ((self._wave_len[freq_channel] / (4 * np.pi * dist)) ** 2) * antenna_gain


class ChannelNetworkModel:
    """Network model for wireless channels
    Graph node represents a communication node, and graph edge represents a channel between them.
    Graph edge contains "channel model" object for calculating and storing a channel gain.
    """
    def __init__(self, channel_model_type: str, center_freq: Union[np.ndarray, List]):
        self._graph = nx.Graph()
        self._channel_model_factory = ChannelModelFactory(channel_model_type, center_freq)

    def add_node(self, node: 'Node'):
        """Add a node to the graph.
        Graph node is represented by a string node name, and node object is set as an attribute.
        Args:
            node: a communication node
        """
        self._graph.add_node(node.name, node=node)

    def add_link(self, link: Tuple['Node', 'Node']):
        """Add a link to the graph.
        Graph edge is created between two nodes and channel model is added as an attribute.
        Args:
            link: a tuple of nodes
        """
        channel_model = self._channel_model_factory.get_channel_model(link)
        self._graph.add_edge(link[0].name, link[1].name, channel_model=channel_model)

    def add_node_and_link(self, node: Union['Node', List['Node']]):
        """Add a node or list of nodes and relevant links to the graph.
        Links are added for all existing nodes potentially communicating or interfering with the given node.
        Args:
            node: communication node or list of communication nodes
        """
        if not isinstance(node, list):
            node = [node]
        for n1 in node:
            self.add_node(n1)
            for gn in self._graph.nodes:
                n2 = self._graph.nodes[gn]['node']
                if n1 != n2 and\
                        (n1.trx_type == 'tx_rx' or n2.trx_type == 'tx_rx' or
                         (n1.trx_type == 'rx' and n2.trx_type == 'tx') or
                         (n1.trx_type == 'tx' and n2.trx_type == 'rx')):
                    self.add_link((n1, n2))

    def get_channel_gain(self, tx_node: 'Node', rx_node: Union['Node', List['Node']], freq_channel: int) \
            -> Union[float, np.ndarray]:
        """Obtain channel gain from one node to the other for the given frequency channel
        If multiple receive nodes are given, the channel gains towards all receive nodes are returned.
        Args:
            tx_node: a transmit node
            rx_node: a receive node (or list of receive nodes)
            freq_channel: frequency channel
        Returns:
            channel gain or an array of channel gains
        """
        if isinstance(rx_node, list):
            rx_node_list = rx_node
        else:
            rx_node_list = [rx_node]
        gain = np.zeros(len(rx_node_list))
        for ind, rn in enumerate(rx_node_list):
            if tx_node.name == rn.name:
                gain[ind] = np.inf
            else:
                gain[ind] = self._graph[tx_node.name][rn.name]['channel_model'].get_channel_gain(freq_channel)
        if not isinstance(rx_node, list):
            gain = gain[0]
        return gain

    def mark_update_needed(self, node_list: List['Node']):
        """Mark that channel gains affected by the movement of a given list of nodes should be updated.
        Args:
            node_list: list of nodes that have moved
        """
        links = self._graph.edges([node.name for node in node_list])
        for link in links:
            self._graph[link[0]][link[1]]['channel_model'].mark_update_needed()
