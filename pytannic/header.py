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
Tannic protocol header utilities.

This module defines the binary header format used for tensor and
metadata transport over networks or in files.

Header layout (little-endian):
    
    [magic: 4 bytes][version: 1 byte][checksum: 2 bytes][nbytes: 8 bytes]

- `magic` : 32-bit unsigned int identifying the protocol (default: ASCII 'ERIC')
- `version` : 8-bit unsigned int protocol version
- `checksum` : 16-bit unsigned int for optional payload verification
- `nbytes` : 64-bit unsigned int size of the payload in bytes
"""

from dataclasses import dataclass
from struct import pack, unpack

# Protocol magic number: ASCII encoded as little-endian 32-bit integer
MAGIC = (69 | (82 << 8) | (73 << 16) | (67 << 24)) 

@dataclass
class Header:
    """
    Binary header for tannic tensor/metadata messages.

    Attributes
    ----------
    magic : int
        Protocol identifier (default: `MAGIC`).
    version : int
        Protocol version number.
    checksum : int
        Optional checksum for payload validation.
    nbytes : int
        Size of the payload in bytes.

    Class Attributes
    ----------------
    FORMAT : str
        Struct format string for packing/unpacking:
        `"<I B H Q"` (little-endian: uint32, uint8, uint16, uint64)
    """
    FORMAT = "<I B H Q"   
    magic: int
    version: int
    checksum: int
    nbytes: int 

    def pack(self) -> bytes:
        """
        Pack the header fields into a binary representation.

        Returns
        -------
        bytes
            Serialized header in little-endian byte order.
        """
        return pack(
            self.FORMAT,
            self.magic,
            self.version,
            self.checksum,
            self.nbytes, 
        )

    @classmethod
    def unpack(cls, data: bytes):
        """
        Deserialize a binary blob into a Header instance.

        Parameters
        ----------
        data : bytes
            Byte string of length equal to `calcsize(Header.FORMAT)`.

        Returns
        -------
        Header
            Header instance with fields populated from `data`.

        Raises
        ------
        struct.error
            If `data` does not match the expected format length.
        """
        unpacked = unpack(cls.FORMAT, data)
        return cls(*unpacked)
      