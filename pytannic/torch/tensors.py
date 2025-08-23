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
Tensor serialization utilities for network transport.

This module defines a compact binary protocol for sending and receiving
PyTorch tensors across a network. A serialized tensor is laid out as:

    [Header][Metadata][Raw Buffer]

- The **Header** is defined in `pytannic.header.Header` and contains
  basic framing information (magic number, version, checksum, payload size).
- The **Metadata** block encodes dtype, shape, and buffer size.
- The **Raw Buffer** contains contiguous tensor bytes in row-major order.

Only CPU tensors are supported. GPU tensors are automatically moved
to CPU before serialization.

Examples
--------
>>> import torch
>>> from pytannic.torch.tensor import serialize, deserialize
>>> x = torch.arange(6, dtype=torch.int32).reshape(2, 3)
>>> data = serialize(x)
>>> y = deserialize(data)
>>> torch.equal(x, y)
True
"""

from dataclasses import dataclass
from struct import calcsize, pack, unpack
from torch import Tensor
from torch import frombuffer
from pytannic.torch.types import dtypeof, dcodeof
from pytannic.header import Header, MAGIC 

@dataclass
class Metadata: 
    """
    Tensor metadata for serialization.

    Attributes
    ----------
    dcode : int
        Encoded dtype (see `pytannic.torch.types.dcodeof`).
    offset : int
        Byte offset into the raw buffer (currently always 0).
    nbytes : int
        Number of bytes in the raw tensor buffer.
    rank : int
        Tensor rank (number of dimensions).
    shape : tuple[int, ...]
        Tensor shape as a tuple of dimension sizes.
    """

    dcode: int
    offset: int
    nbytes: int
    rank: int
    shape: tuple[int, ...]  
 
    def pack(self) -> bytes:
        """
        Serialize the metadata into a binary blob.

        Returns
        -------
        bytes
            Packed metadata using little-endian struct format.
        """
        return pack(self.format, self.dcode, self.offset, self.nbytes, self.rank, *self.shape)

    @classmethod
    def unpack(cls, data: bytes):  
        """
        Deserialize a `Metadata` instance from binary data.

        Parameters
        ----------
        data : bytes
            Binary blob containing packed metadata.

        Returns
        -------
        Metadata
            A new `Metadata` instance.
        """
        head = calcsize("<B Q Q B")
        dcode, offset, nbytes, rank = unpack("<B Q Q B", data[:head])
        if rank == 0:
            shape = ()
        else:
            shape = unpack(f"<{rank}Q", data[head:head + rank * 8])
        return cls(dcode, offset, nbytes, rank, shape) 

    @property
    def format(self) -> str: 
        """
        Struct format string for packing/unpacking tensor metadata.

        The format is `"<B Q Q B{rank}Q"`, which corresponds to the
        following C struct layout in little-endian order (no padding):

        .. code-block:: cpp

            struct Metadata<Tensor> {
                uint8_t dcode;        // 1 byte
                size_t offset;  // 8 bytes
                size_t nbytes;  // 8 bytes
                uint8_t rank;         // 1 byte
                size_t shape[rank]; // 8 bytes each
            };

        Layout (before shape):
        ----------------------
        - dcode  : 1 byte
        - offset : 8 bytes
        - nbytes : 8 bytes
        - rank   : 1 byte
        --------------------------------
        Fixed size = 18 bytes + (8 * rank) for shape

        Shape array:
        ------------
        - Each dimension size is stored as an `unsigned long long` (8 bytes).
        - The number of entries equals `rank`.

        Returns
        -------
        str
            A format string of the form `"<B Q Q B{rank}Q"`.
        """ 
        return f"<B Q Q B{self.rank}Q"
     

def serialize(tensor: Tensor) -> bytes:
    """
    Serialize a PyTorch tensor into a binary blob for network transport.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor. If the tensor is on a GPU, it will be moved to CPU.

    Returns
    -------
    bytes
        A binary blob representing the tensor, consisting of:

        - Header
        - Metadata
        - Raw buffer (tensor bytes)

    Notes
    -----
    - Only contiguous CPU tensors are supported.
    - Dtype is encoded using `dcodeof`.
    """
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    rank = tensor.dim()
    shape  = tensor.shape
    buffer = tensor.numpy().tobytes() 
    metadata = Metadata(dcode=dcodeof(tensor.dtype), offset=0, nbytes=len(buffer), rank=rank, shape=shape)  
    header = Header(magic=MAGIC, version=1, checksum=0xABCD , nbytes = calcsize(metadata.format) + len(buffer)) 
    return header.pack() + metadata.pack()+ buffer 


def deserialize(data: bytes) -> Tensor:
    """
    Deserialize a PyTorch tensor from a binary blob.

    Parameters
    ----------
    data : bytes
        A binary blob produced by `serialize`.

    Returns
    -------
    torch.Tensor
        The reconstructed tensor with the same dtype and shape.

    Raises
    ------
    ValueError
        If the dcode in metadata is not recognized.

    Notes
    -----
    - Uses `torch.frombuffer`, so the returned tensor shares memory
      with the input `data` buffer when possible.
    """
    hsize = calcsize(Header.FORMAT) 
    head = calcsize("<B Q Q B")
    dcode, offset, nbytes, rank = unpack("<B Q Q B", data[hsize : hsize + head])
    
    msize = head + 8 * rank
    metadata = Metadata.unpack(data[hsize : hsize + msize]) 
    offset = hsize + msize
    buffer = bytearray(data[offset: offset + nbytes])
    return frombuffer(buffer, dtype=dtypeof(dcode)).reshape(metadata.shape) 