from typing import TYPE_CHECKING, Optional, Callable, List, Union
import simpy
import numpy as np
if TYPE_CHECKING:
    from arena_server.simulator import Simulator
    from arena_server.node import Node
    from arena_server.packet import PhyPacket


class RadioSignal:
    """ RadioSignal object describes the radio signal transmitted and received over the air
    """
    def __init__(self, start_time: float, end_time: float, freq_channel: int, tx_node: Optional['Node'],
                 rx_node: Optional[Union['Node', List['Node']]], tx_power: float, packet: Optional['PhyPacket'],
                 callback: Callable):
        """
        Args:
            start_time: start time of the radio signal
            end_time: end time of the radio signal
            freq_channel: frequency channel of the radio signal
            tx_node: transmit node of the radio signal
            rx_node: receive node of the radio signal.
                     If there are multiple receive nodes, it is a list of receive nodes.
            tx_power: transmit power of the radio signal
            packet: packet object if the radio signal is from the packet transmission
            callback: callback function executed after the radio signal is over
        """
        self._start_time = start_time
        self._end_time = end_time
        self._freq_channel = freq_channel
        self._tx_node = tx_node
        self._rx_node = rx_node
        self._tx_power = tx_power
        self._packet = packet
        self._callback = callback
        self._target_rx_power = 0
        self._sum_non_target_rx_energy: np.array = np.zeros(1)
        if isinstance(rx_node, list):
            self._target_rx_power = [0] * len(rx_node)
            self._sum_non_target_rx_energy = np.zeros(len(rx_node))

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def end_time(self) -> float:
        return self._end_time

    @property
    def freq_channel(self) -> int:
        return self._freq_channel

    @property
    def tx_node(self) -> 'Node':
        return self._tx_node

    @property
    def rx_node(self) -> 'Node':
        return self._rx_node

    @property
    def tx_power(self) -> float:
        return self._tx_power

    @property
    def callback(self) -> Callable:
        return self._callback

    @property
    def packet(self) -> 'PhyPacket':
        return self._packet

    def overlap_time(self, radio_signal: 'RadioSignal') -> float:
        """ Calculate the overlap time between two radio signals
        Args:
            radio_signal: the radio signal with which the overlap time is calculated
        Returns:
            overlap time
        """
        start = max(self.start_time, radio_signal.start_time)
        end = min(self.end_time, radio_signal.end_time)
        return max(end - start, 0)

    def set_target_rx_power(self, target_rx_power: Union[float, List[float]]):
        """ Set target receive power.
        Args:
            target_rx_power: Receive power from the intended radio signal source.
        """
        self._target_rx_power = target_rx_power

    def get_target_rx_power(self) -> Union[float, List[float]]:
        """ Get target receive power.
        Returns:
            target_rx_power: Receive power from the intended radio signal source.
        """
        return self._target_rx_power

    def add_non_target_rx_energy(self, energy: Union[float, List[float]]):
        """ Add non-target receive energy
        Args:
            energy: energy from the unintended source accumulated during the overlap time.
        """
        if isinstance(energy, list):
            energy = np.array(energy)
        self._sum_non_target_rx_energy += energy

    def get_avg_non_target_rx_power(self) -> Union[float, List[float]]:
        """ Get non-target average receive power
        Returns:
            Average power from the unintended sources during the radio signal reception.
        """
        avg_power = list(self._sum_non_target_rx_energy / (self._end_time - self._start_time))
        if len(avg_power) == 1:
            avg_power = avg_power[0]
        return avg_power


class RadioEnvironment:
    """ Radio environment where radio signals are transmitted and received.
    """
    def __init__(self, simulator: 'Simulator', num_freq_channel: int):
        """
        Args:
            simulator: simulator
            num_freq_channel: number of frequency channels
        """
        self._simulator = simulator
        # radio_signal_list is the list of active radio signals for each frequency channel.
        self._radio_signal_list: List[List[RadioSignal]] = []
        for _ in range(num_freq_channel):
            self._radio_signal_list.append([])
        self._num_freq_channel = num_freq_channel

    def send_packet(self, packet: 'PhyPacket') -> simpy.Timeout:
        """ Send packet to the radio environment
        Args:
            packet: packet to be transmitted.
        Returns:
            Timeout event which expires when the packet transmission is over.
        """
        now = self._simulator.now()
        time_duration = packet.time_duration
        freq_channel = packet.freq_channel
        tx_node = packet.tx_node
        rx_node = packet.rx_node
        tx_power = packet.tx_power
        # Radio signal starts from now and exists during time_duration. When it is over, self._receive_packet is called.
        radio_signal = RadioSignal(now, now + time_duration, freq_channel, tx_node, rx_node, tx_power,
                                   packet, self._receive_packet)
        return self._start_radio_signal(radio_signal)

    @staticmethod
    def _receive_packet(radio_signal: RadioSignal):
        """ Receive packet from the radio environment. This function is called when packet transmission is over.
        Args:
            radio_signal: radio signal carrying the packet
        """
        packet = radio_signal.packet
        rx_node = packet.rx_node
        rx_power = radio_signal.get_target_rx_power()  # Get receive power of the packet from the radio signal
        interference = radio_signal.get_avg_non_target_rx_power()  # Get interference power from the radio signal
        if not isinstance(packet.rx_node, list):
            rx_node = [rx_node]
            rx_power = [rx_power]
            interference = [interference]
        for rn, rxp, interf in zip(rx_node, rx_power, interference):
            # Deliver packet to the intended receivers with receive and interference power
            rn.receive_phy_packet(packet, rxp, interf)

    def start_sensing(self, node: 'Node', time_duration: float, freq_channel: int) -> simpy.Timeout:
        """ Start sensing
        Args:
            node: node conducting sensing
            time_duration: time duration of sensing
            freq_channel: frequency channel of sensing
        Returns:
            Timeout event which expires when the sensing is over.
        """
        now = self._simulator.now()
        # radio signal is generated with only receive node.
        # self._finish_sensing is called when the radio signal is over.
        radio_signal = RadioSignal(now, now + time_duration, freq_channel, None, node, 0, None, self._finish_sensing)
        return self._start_radio_signal(radio_signal)

    @staticmethod
    def _finish_sensing(radio_signal: RadioSignal):
        """ Finish sensing. This function is called when the sensing is over.
        Args:
            radio_signal: radio signal having the sensing result
        """
        # sensed power is the average non-target receive power
        sensed_power = radio_signal.get_avg_non_target_rx_power()
        sensed_power_dbm = 10 * np.log10(sensed_power) if sensed_power > 0 else -np.inf
        node = radio_signal.rx_node
        freq_channel = radio_signal.freq_channel
        time_duration = radio_signal.end_time - radio_signal.start_time
        node.finish_sensing(sensed_power_dbm, time_duration, freq_channel)

    def _accumulate_non_target_receive_energy(self, tx_radio_signal: 'RadioSignal', rx_radio_signal: 'RadioSignal'):
        """ Accumulate non-target receive energy from the transmit radio signal to the receive radio signal
        Args:
            tx_radio_signal: transmit radio signal
            rx_radio_signal: receive radio signal
        """
        overlap_time = tx_radio_signal.overlap_time(rx_radio_signal)
        freq_channel = tx_radio_signal.freq_channel
        tx_node = tx_radio_signal.tx_node
        rx_node = rx_radio_signal.rx_node
        if (tx_node is not None) and (rx_node is not None) and (freq_channel == rx_radio_signal.freq_channel):
            if not isinstance(rx_node, list):
                rx_node = [rx_node]
            rx_energy = []
            for rn in rx_node:
                channel_gain = self._simulator.channel_network_model.get_channel_gain(tx_node, rn, freq_channel)
                rx_energy.append(channel_gain * tx_radio_signal.tx_power * overlap_time)
                if rx_radio_signal.packet is not None:
                    tx_signal_rx_node = tx_radio_signal.rx_node
                    if isinstance(tx_signal_rx_node, list):
                        tx_signal_rx_node = tx_signal_rx_node[0]
                    self._simulator.logger.logging({'type': 'interference', 'overlap time': overlap_time,
                                                    'freq channel': freq_channel,
                                                    'tx packet': [tx_node.name, tx_signal_rx_node.name],
                                                    'rx packet': [rx_radio_signal.tx_node.name, rn.name],
                                                    'interference energy': rx_energy})
            if len(rx_energy) == 1:
                rx_energy = rx_energy[0]
            rx_radio_signal.add_non_target_rx_energy(rx_energy)

    def _start_radio_signal(self, radio_signal: 'RadioSignal') -> simpy.Timeout:
        """ Start radio signal
        Args:
            radio_signal: radio signal that will be started
        Returns:
            Timeout event which expires when the radio signal is over.
        """
        # Check out all existing radio signals on the same frequency channel
        freq_channel = radio_signal.freq_channel
        for sig in self._radio_signal_list[freq_channel]:
            # Accumulate non-target receive energy to the existing radio signal
            if radio_signal.tx_node is not None and sig.rx_node is not None:
                self._accumulate_non_target_receive_energy(radio_signal, sig)
            # Get non-target receive energy from the existing radio signal
            if radio_signal.rx_node is not None and sig.tx_node is not None:
                self._accumulate_non_target_receive_energy(sig, radio_signal)
        # Compute the target receive power of the radio signal
        if radio_signal.tx_node is not None and radio_signal.rx_node is not None:
            rx_node = radio_signal.rx_node
            if not isinstance(rx_node, list):
                rx_node = [rx_node]
            rx_power = []
            for rn in rx_node:
                channel_gain = self._simulator.channel_network_model.get_channel_gain(radio_signal.tx_node,
                                                                                      rn, freq_channel)
                rx_power.append(channel_gain * radio_signal.tx_power)
            if not isinstance(radio_signal.rx_node, list):
                rx_power = rx_power[0]
            radio_signal.set_target_rx_power(rx_power)
        self._radio_signal_list[freq_channel].append(radio_signal)
        delay = radio_signal.end_time - self._simulator.now()
        timeout_event = self._simulator.get_timeout_event(delay, radio_signal)
        # Call _finish_radio_signal when the radio signal is over
        timeout_event.callbacks.append(self._finish_radio_signal)
        return timeout_event

    def _finish_radio_signal(self, event: simpy.Timeout):
        """ Finish radio signal
        Remove radio signal from the active radio signal list, and call the callback of the radio signal.
        Args:
            event: time-out event triggering the end of the radio signal
        """
        radio_signal = event.value
        freq_channel = radio_signal.freq_channel
        self._radio_signal_list[freq_channel].remove(radio_signal)
        radio_signal.callback(radio_signal)
