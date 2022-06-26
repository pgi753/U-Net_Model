from typing import TYPE_CHECKING, Callable, List, Dict
if TYPE_CHECKING:
    from arena_server.simulator import Simulator


class SimulationLogger:
    def __init__(self, simulator: 'Simulator'):
        self._clock_interval = 0
        self._log_count = 0
        self._simulator = simulator
        self._handler_list: List[Callable] = []

    def set_clock(self, clock_interval: float):
        self._clock_interval = clock_interval
        self._simulator.register_process(self._run_clock())

    def _run_clock(self):
        while True:
            self.logging({'type': 'clock'})
            time_out_event = self._simulator.get_timeout_event(self._clock_interval)
            yield time_out_event

    def add_handler(self, handler: Callable):
        self._handler_list.append(handler)

    def add_file_handler(self, file_name: str):
        f = open(file_name, 'w')
        self.add_handler(lambda x: f.write(f'{x}\n'))

    def add_print_handler(self):
        self.add_handler(lambda x: print(f'{x}'))

    def logging(self, log: Dict):
        time_log = {'time': self._simulator.now()}
        time_log.update(log)
        for handler in self._handler_list:
            handler(time_log)
