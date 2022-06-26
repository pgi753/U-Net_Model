from typing import TYPE_CHECKING, Dict, Optional
if TYPE_CHECKING:
    from arena_server.network_operator import NetworkOperator
    from utility.messenger import Messenger


class Controller:
    def __init__(self):
        self._network_operator: Optional['NetworkOperator'] = None

    def set_network_operator(self, network_operator: 'NetworkOperator'):
        self._network_operator = network_operator

    def request_decision(self, observation: Dict) -> Dict:
        raise NotImplementedError


class RemoteController(Controller):
    def __init__(self):
        super(Controller, self).__init__()
        self._messenger: Optional['Messenger'] = None

    def set_messenger(self, messenger: 'Messenger'):
        self._messenger = messenger

    def request_decision(self, observation: Dict) -> Optional[Dict]:
        self._messenger.send('observation', observation)
        type, msg = self._messenger.recv()
        if type == 'action':
            return msg
        else:
            return None
