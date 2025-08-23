# Copyright 2025 Eric Hermosis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You can obtain a copy of the License at:
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This software is distributed "AS IS," without warranties or conditions.
# See the License for specific terms. 


"""
TCP client for tensor transport.

This class provides a thin wrapper around Python's `socket` for sending
and receiving messages that follow the **tannic tensor protocol**:

    [Header][Payload]

- **Header** is defined in `pytannic.header.Header`. It includes:
  - magic number (`MAGIC`) for validation
  - protocol version
  - checksum (not validated yet)
  - payload size in bytes
- **Payload** is application-dependent:
  - Serialized tensor (see `pytannic.torch.tensor`)
  - Serialized metadata (see `pytannic.torch.parameters`)
  - Or other binary blobs

The client ensures full reception of both the header and payload.
"""

import socket
from struct import calcsize 
from pytannic.header import MAGIC, Header

class Client:
    """
    TCP client for communicating with a tannic tensor server.

    Supports context-manager usage:

    >>> with Client("localhost", 9000) as client:
    ...     client.send(b"hello")
    ...     reply = client.receive()

    Attributes
    ----------
    host : str
        Server hostname or IP.
    port : int
        Server TCP port.
    socket : socket.socket or None
        Underlying socket object. Created in `begin()`.
    """
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None

    def __enter__(self):
        """Enable `with Client(...) as client:` usage."""
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the socket when leaving context manager scope."""
        self.close()

    def begin(self):
        """
        Establish a TCP connection to the server.

        Raises
        ------
        OSError
            If the socket cannot connect to the given host/port.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def close(self):
        """
        Close the socket connection if open.
        """
        if self.socket:
            self.socket.close()
            self.socket = None

    def send(self, data: bytes): 
        """
        Send raw bytes to the server.

        Parameters
        ----------
        data : bytes
            A complete tannic message, including header and payload.

        Raises
        ------
        RuntimeError
            If the socket is not connected.
        """
        if not self.socket:
            raise RuntimeError("Socket not connected")
        self.socket.sendall(data)
 
    def receive(self) -> bytes: 
        """
        Receive a complete tannic message (header + payload).

        Returns
        -------
        bytes
            The raw message received, including the header and payload.

        Raises
        ------
        ValueError
            If the received header has an invalid magic number.
        ConnectionError
            If the socket closes before the expected number of bytes is received.
        """
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