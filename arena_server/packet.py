from typing import TYPE_CHECKING, List, Dict, Optional, Union
if TYPE_CHECKING:
    from arena_server.node import Node


class UpperLayerPacket:
    def __init__(self, type: str, subtype: Optional[str] = None, sequence_number: int = 0, data_length: int = 0,
                 control_info: Optional[Dict] = None):
        self._type = type
        self._subtype = subtype
        self._sequence_number = sequence_number
        self._data_length = data_length
        self._control_info = control_info

    @property
    def type(self):
        return self._type

    @property
    def subtype(self):
        return self._subtype

    @property
    def sequence_number(self):
        return self._sequence_number

    @property
    def data_length(self):
        return self._data_length

    @property
    def control_info(self):
        return self._control_info


class PhyPacket:
    def __init__(self, tx_node: 'Node', rx_node: Union['Node', List['Node']], time_duration: float,
                 freq_channel: int, tx_power: float, sinr_threshold: float, upper_layer_packet: UpperLayerPacket):
        self._tx_node = tx_node
        self._rx_node = rx_node
        self._time_duration = time_duration
        self._freq_channel = freq_channel
        self._tx_power = tx_power
        self._sinr_threshold = sinr_threshold
        self._upper_layer_packet = upper_layer_packet
        self._start_time = -1
        self._sinr = -1

    @property
    def tx_node(self) -> 'Node':
        return self._tx_node

    @property
    def rx_node(self) -> Union['Node', List['Node']]:
        return self._rx_node

    @property
    def time_duration(self) -> float:
        return self._time_duration

    @property
    def freq_channel(self) -> int:
        return self._freq_channel

    @property
    def tx_power(self) -> float:
        return self._tx_power

    @property
    def sinr_threshold(self) -> float:
        return self._sinr_threshold

    @property
    def upper_layer_packet(self) -> UpperLayerPacket:
        return self._upper_layer_packet

    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, start_time: float):
        self._start_time = start_time

    @property
    def sinr(self) -> float:
        return self._sinr

    @sinr.setter
    def sinr(self, sinr: float):
        self._sinr = sinr
