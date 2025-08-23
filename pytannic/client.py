import socket
from struct import calcsize 
from pytannic.header import MAGIC, Header

class Client:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def begin(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None

    def send(self, data: bytes): 
        if not self.socket:
            raise RuntimeError("Socket not connected")
        self.socket.sendall(data)
 
    def receive(self) -> bytes: 
        hsize = calcsize(Header.FORMAT)
        header_data = self._recvall(hsize)
        header = Header.unpack(header_data)
        if header.magic != MAGIC:
            raise ValueError("Invalid magic number in received data")
  
        payload = self._recvall(header.nbytes) 
        return header_data + payload

    def _recvall(self, size: int) -> bytes: 
        buffer = b""
        while len(buffer) < size:
            chunk = self.socket.recv(size - len(buffer))
            if not chunk:
                raise ConnectionError("Socket closed before receiving enough data")
            buffer += chunk
        return buffer