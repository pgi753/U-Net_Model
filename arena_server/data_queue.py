import numpy as np
from simpy import Event
from typing import TYPE_CHECKING, Generator, Any, Optional
if TYPE_CHECKING:
    from arena_server.simulator import Simulator
    from arena_server.node import Node


class DataQueue:
    """ Data queue class
    """
    def __init__(self, name: str):
        """
        Args:
            name: name of the data queue
        """
        self._name: str = name
        self._node: Optional[Node] = None
        self._simulator: Optional['Simulator'] = None

    @property
    def name(self):
        return self._name

    def set_node(self, node: 'Node'):
        self._node = node

    def set_simulator(self, simulator: 'Simulator'):
        self._simulator = simulator

    @property
    def node(self) -> 'Node':
        return self._node

    def deque(self, *args):
        """ Get data out of the queue
        """
        raise NotImplementedError

    def enque(self, *args):
        """ Put data into the queue
        """
        raise NotImplementedError

    def get_queue_size(self):
        """
        Returns:
            Queue size: the number of data bytes in the queue
        """
        raise NotImplementedError

    def get_total_data_in(self):
        """
        Returns:
            Total data in: the accumulated number of data bytes put into the queue
        """
        raise NotImplementedError

    def get_total_data_out(self):
        """
        Returns:
            Total data out: the accumulated number of data bytes taken out of the queue
        """
        raise NotImplementedError


class InfiniteDataSource(DataQueue):
    """ Infinite data source
    The amount of data in the queue is infinite and the number of output bytes are counted when data is dequeued.
    """
    def __init__(self, name: str):
        super(InfiniteDataSource, self).__init__(name)
        self._total_data_out: int = 0

    def deque(self, num_bytes: int):
        """
        Args:
            num_bytes: the number of data bytes that will be dequeued.
        Returns:
            data_out: actual amount of data bytes that is dequeued.
        """
        self._total_data_out += num_bytes
        self._simulator.logger.logging({'type': 'deque', 'name': self.name, 'num bytes': num_bytes,
                                        'total data out': self._total_data_out})
        data_out = num_bytes
        return data_out

    def enque(self):
        pass

    def get_queue_size(self) -> int:
        return np.infty

    def get_total_data_in(self) -> int:
        return self._total_data_out

    def get_total_data_out(self) -> int:
        return self._total_data_out


class PoissonDataSource(DataQueue):
    """ Poisson data source
    Data arrival time is spaced by the exponential distribution with mean avg_inter_arrival_time.
    Arrival data amount is fixed to arrival_data_size (byte).
    Arrival data is dropped if the queue size is over the maximum queue size.
    """
    def __init__(self, name: str, max_queue_size: int, avg_inter_arrival_time: float, arrival_data_size: int):
        """
        Args:
            name: name of the queue
            max_queue_size: maximum queue size (byte)
            avg_inter_arrival_time: average inter-arrival time (ms) between packet arrival
            arrival_data_size: the amount of data (byte) that arrives at each arrival event
        """
        super(PoissonDataSource, self).__init__(name)
        self._max_queue_size: int = max_queue_size
        self._avg_inter_arrival_time: float = avg_inter_arrival_time
        self._arrival_data_size: int = arrival_data_size
        self._queue_size: int = 0  # current queue size (byte)
        self._total_dropped_data: int = 0  # dropped data due to full queue (byte)
        self._total_data_in: int = 0
        self._total_data_out: int = 0

    def deque(self, num_bytes: int):
        """
        Args:
            num_bytes: the number of data bytes that will be dequeued.
        Returns:
            data_out: actual amount of data bytes that is dequeued.
        """
        data_out = min(self._queue_size, num_bytes)
        self._queue_size -= data_out
        self._total_data_out += data_out
        self._simulator.logger.logging({'type': 'deque', 'name': self.name, 'num bytes': num_bytes,
                                        'data out': data_out, 'total data out': self._total_data_out,
                                        'queue size': self._queue_size})
        return data_out

    def enque(self, num_bytes: int):
        """
        Args:
            num_bytes: the number of data bytes that will be enqueued.
        """
        queue_remaining_space = self._max_queue_size - self._queue_size
        data_in = min(queue_remaining_space, num_bytes)
        dropped_data = num_bytes - data_in
        self._total_dropped_data += dropped_data
        self._queue_size += data_in
        self._total_data_in += data_in
        self._simulator.logger.logging({'type': 'data arrival', 'name': self.name, 'num bytes': num_bytes,
                                        'data in': data_in, 'total data in': self._total_data_in,
                                        'dropped data:': dropped_data, 'total dropped data': self._total_dropped_data,
                                        'queue size': self._queue_size})

    def get_queue_size(self) -> int:
        return self._queue_size

    def get_total_data_in(self) -> int:
        return self._total_data_in

    def get_total_data_out(self) -> int:
        return self._total_data_out

    def set_simulator(self, simulator: 'Simulator'):
        super(PoissonDataSource, self).set_simulator(simulator)
        simulator.register_process(self._run())

    def _run(self) -> Generator[Event, Any, Any]:
        """ DES generator for queue.
        enque is executed at events spaced by exponentially distributed inter-arrival time.
        """
        while True:
            self.enque(self._arrival_data_size)
            inter_arrival_time = np.random.exponential(self._avg_inter_arrival_time)
            event = self._simulator.get_timeout_event(inter_arrival_time)
            yield event


class DataSink(DataQueue):
    def __init__(self, name: str):
        super(DataSink, self).__init__(name)
        self._total_data_in: int = 0

    def deque(self):
        pass

    def enque(self, num_bytes: int):
        """
        Args:
            num_bytes: the number of data bytes that will be enqueued.
        """
        self._total_data_in += num_bytes
        self._simulator.logger.logging({'type': 'enque', 'name': self.name, 'num bytes': num_bytes,
                                        'total data in': self._total_data_in})

    def get_queue_size(self):
        return 0

    def get_total_data_in(self) -> int:
        return self._total_data_in

    def get_total_data_out(self) -> int:
        return self._total_data_in
