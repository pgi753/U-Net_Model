import socket
from utility.messenger import Messenger
from typing import Optional, Dict


class Player:
    def __init__(self, identifier: str):
        self._identifier = identifier
        self._server_address: str = ''
        self._server_port: int = 0
        self._socket: socket = None
        self._messenger: Optional[Messenger] = None
        self._operator_info: Optional[Dict] = None
        self._first_step = True

    @property
    def operator_info(self):
        return self._operator_info

    def connect_to_server(self, server_address: str, server_port: int):
        self._server_address = server_address
        self._server_port = server_port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._server_address, self._server_port))
        self._messenger = Messenger(self._socket)
        self._messenger.send('identifier', self._identifier)
        type, msg = self._messenger.recv()
        if type == 'operator_info':
            self._operator_info = msg

    def step(self, action: Dict) -> Optional[Dict]:
        if self._first_step:
            self._messenger.recv()  # Discard the first observation
            self._first_step = False
        self._messenger.send('action', action)
        type, msg = self._messenger.recv()
        if type == 'observation':
            return msg
        else:
            return None
