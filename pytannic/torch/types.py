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
dtype serialization utilities for tensor clients.

This module provides a stable mapping between PyTorch `torch.dtype`
objects and compact integer codes ("dcodes"). These codes can be used
to serialize tensor metadata for network protocols, file formats,
or any client-server interaction involving tensors.

Mappings
--------
    int8        <-> 12
    int16       <-> 13
    int32       <-> 14
    int64       <-> 15
    float16     <-> 23
    bfloat16    <-> 231
    float32     <-> 24
    float64     <-> 25
    complex64   <-> 37
    complex128  <-> 38

Example
-------
>>> from torch import float32, int64
>>> from dtypes import dcodeof, dtypeof
>>> dcodeof(float32)
24
>>> dtypeof(15)
torch.int64
"""

from torch import dtype
from torch import (
    int8,
    int16,
    int32,
    int64,
    float16,
    bfloat16,
    float32,
    float64,
    complex64,
    complex128,
)

def dcodeof(dtype: dtype):
    """
    Get the integer serialization code ("dcode") for a given `torch.dtype`.

    Parameters
    ----------
    dtype : torch.dtype
        The PyTorch dtype to encode.

    Returns
    -------
    int
        The corresponding integer code. Returns `0` if the dtype is unsupported.

    Examples
    --------
    >>> dcodeof(torch.float32)
    24
    >>> dcodeof(torch.int16)
    13
    >>> dcodeof(torch.bool)
    0  # not supported
    """

    if dtype == int8:
        return 12
    elif dtype == int16:
        return 13
    elif dtype == int32:
        return 14
    elif dtype == int64:
        return 15
    elif dtype == float16:
        return 231
    elif dtype == float16:
        return 23
    elif dtype == float32:
        return 24
    elif dtype == float64:
        return 25
    elif dtype == complex64:
        return 37
    elif dtype == complex128:
        return 38
    else:
        return 0     
    
def dtypeof(code: int) -> dtype:

    """
    Get the `torch.dtype` corresponding to a serialization code ("dcode").

    Parameters
    ----------
    code : int
        The integer code to decode.

    Returns
    -------
    torch.dtype
        The corresponding dtype.

    Raises
    ------
    ValueError
        If the code is not recognized.

    Examples
    --------
    >>> dtypeof(24)
    torch.float32
    >>> dtypeof(15)
    torch.int64
    >>> dtypeof(99)
    Traceback (most recent call last):
        ...
    ValueError: Unknown code
    """
    
    if code == 12:
        return int8
    elif code == 13:
        return int16
    elif code == 14:
        return int32
    elif code == 15:
        return int64
    elif code == 23:
        return float16
    elif code == 231:
        return bfloat16
    elif code == 24:
        return float32
    elif code == 25:
        return float64
    elif code == 37:
        return complex64
    elif code == 38:
        return complex128
    else:
        raise ValueError("Unknown code")