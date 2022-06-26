import socket
from utility.messenger import Messenger
from typing import TYPE_CHECKING, Optional, List, Dict
from arena_server.controller import RemoteController
if TYPE_CHECKING:
    from arena_server.network_operator import NetworkOperator


class RemotePlayer:
    def __init__(self, identifier: str = ''):
        self._identifier = identifier
        self._messenger: Optional['Messenger'] = None
        self._remote_controller = RemoteController()
        self._network_operator: Optional['NetworkOperator'] = None

    @property
    def identifier(self):
        return self._identifier

    @property
    def messenger(self):
        return self._messenger

    def set_remote_connection(self, messenger: Messenger):
        self._messenger = messenger
        self._remote_controller.set_messenger(self._messenger)

    def set_network_operator(self, network_operator: 'NetworkOperator'):
        self._network_operator = network_operator
        self._network_operator.set_controller(self._remote_controller)


class RemotePlayerManager:
    def __init__(self):
        self._server_address: str = ''
        self._server_port: int = 0
        self._server_socket = None
        self._remote_player_list: List[RemotePlayer] = []

    @property
    def remote_player_list(self):
        return self._remote_player_list

    def set_server_socket(self, server_address: str, server_port: int):
        self._server_address = server_address
        self._server_port = server_port
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self._server_address, self._server_port))
        self._server_socket.listen()

    def connect_to_players(self, id_operator_mapping: Dict[str, 'NetworkOperator']):
        while id_operator_mapping:
            client_socket, client_address = self._server_socket.accept()
            messenger = Messenger(client_socket)
            type, identifier = messenger.recv()
            if type == 'identifier' and (identifier in id_operator_mapping):
                remote_player = RemotePlayer(identifier)
                remote_player.set_remote_connection(messenger)
                network_operator = id_operator_mapping.pop(identifier)
                remote_player.set_network_operator(network_operator)
                operator_info = network_operator.get_information()
                remote_player.messenger.send('operator_info', operator_info)
                self._remote_player_list.append(remote_player)
            else:
                client_socket.close()