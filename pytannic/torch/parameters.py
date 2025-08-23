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
Parameter metadata serialization.

Unlike tensor payloads (which are typically transmitted over the network),
parameters are usually **stored in files** (e.g., model checkpoints).
This module defines a binary metadata structure for describing
individual named tensor parameters.

The binary layout is:

    [dcode (1B)][offset (8B)][nbytes (8B)][namelength (4B)][name (bytes)]

Attributes
----------
- **dcode** : dtype code (see `pytannic.torch.types.dcodeof`)
- **offset** : byte offset to parameter data within the file
- **nbytes** : number of bytes of parameter data
- **namelength** : length of the name in bytes
- **name** : parameter name (string)

Examples
--------
>>> m = Metadata(dcode=24, offset=128, nbytes=4096, namelength=4, name="fc1")
>>> blob = m.pack()
>>> Metadata.unpack(blob)
Metadata(dcode=24, offset=128, nbytes=4096, namelength=4, name='')
"""

from dataclasses import dataclass
from struct import pack, unpack, calcsize

@dataclass
class Metadata:
    """
    Metadata structure for a named parameter.

    Attributes
    ----------
    dcode : int
        Encoded dtype for the parameter (see `pytannic.torch.types.dcodeof`).
    offset : int
        Byte offset in the file where the parameter data begins.
    nbytes : int
        Size of the parameter data in bytes.
    namelength : int
        Length of the parameter name in bytes.
    name : str
        Parameter name. Currently not packed/unpacked automatically.
    """
    dcode: int
    offset: int
    nbytes: int
    namelength: int  
    name: str

    FORMAT = "<B Q Q I"   
    def pack(self) -> bytes: 
        """
        Serialize metadata into a binary blob.

        Returns
        -------
        bytes
            Packed metadata (excluding the parameter name).
        """
        return pack(self.FORMAT, self.dcode, self.offset, self.nbytes, self.namelength)

    @classmethod
    def unpack(cls, data: bytes):
        """
        Deserialize metadata from a binary blob.

        Parameters
        ----------
        data : bytes
            Raw binary data containing packed metadata.

        Returns
        -------
        Metadata
            A new `Metadata` instance. The `name` field is left empty,
            since it must be read separately from the stream.
        """
        fsize = calcsize(cls.FORMAT)
        dcode, offset, nbytes, namelength = unpack(cls.FORMAT, data[:fsize])
        return cls(dcode=dcode, offset=offset, nbytes=nbytes, namelength=namelength)

    @property
    def format(self) -> str:
        """
        Struct format string for this metadata.

        The format is `"<B Q Q I"`, which corresponds to the following
        C struct layout in little-endian order (no padding):

        .. code-block:: cpp
        
        struct Metadata<nn::Parameter> {
            uint8_t dcode;        // 1 byte
            size_t offset;        // 8 bytes
            size_t long nbytes;   // 8 bytes
            uint16_t namelength;  // 4 bytes 
        };

        Layout (before name):
        ---------------------
        - dcode      : 1 byte
        - offset     : 8 bytes
        - nbytes     : 8 bytes
        - namelength : 4 bytes
        --------------------------------
        Total        : 21 bytes + name

        Returns
        -------
        str
            Always `"<B Q Q I"`.
        """
        return self.FORMAT  