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
Module serialization for checkpointing.

This module provides utilities to serialize a PyTorch `torch.nn.Module`
into two companion binary files:

1. **Weights file** (`.tannic`)  
   Contains raw parameter tensors stored back-to-back.

   Layout:
       [Header][TensorBytes...]

   - `Header` = `pytannic.header.Header`
   - Each parameter is written as a raw contiguous CPU buffer
     in the order returned by `state_dict()`.

2. **Metadata file** (`.metadata.tannic`)  
   Contains parameter descriptors (dtype, offset, name, etc.).

   Layout:
       [Header][Metadata1][Name1][Metadata2][Name2]...

   - `Header` = `pytannic.header.Header`
   - Each `Metadata` = `pytannic.torch.parameters.Metadata`
   - `Name` is stored as UTF-8 with length `namelength`

These two files together allow reconstructing the module parameters.

Notes
-----
- Parameters are always written in **CPU memory order**.
- Gradients are not serialized (only `.data`).
- Offsets in metadata are relative to the weights file.
"""

from struct import pack
from struct import calcsize
from pathlib import Path  
from torch.nn import Module
from torch.nn import Parameter
from pytannic.header import Header, MAGIC
from pytannic.torch.types import dcodeof
from pytannic.torch.parameters import Metadata

def write(module: Module, filename: str) -> None:
    """
    Serialize a `torch.nn.Module` into `.tannic` files.

    Parameters
    ----------
    module : torch.nn.Module
        The PyTorch module to serialize. Only parameters from
        `module.state_dict()` are saved.
    filename : str
        Base filename for output. Two files are written:

        - `<stem>.tannic` for raw parameter data
        - `<stem>.metadata.tannic` for metadata

    File Layout
    -----------
    **Weights file (`.tannic`):**

    .. code-block:: cpp 
    
        struct Header header;
        unsigned char buffer[];   // concatenated tensor data

    **Metadata file (`.metadata.tannic`):**

    .. code-block:: cpp

        struct Header header;
        struct Metadata<nn::Parameter> {
            uint8_t dcode;        // 1 byte
            size_t offset;        // 8 bytes
            size_t long nbytes;   // 8 bytes
            uint16_t namelength;  // 4 bytes 
        }; 

    Notes
    -----
    - The two files must always be kept together.
    - The header `nbytes` field includes only payload size
      (not counting the header itself).
    """
    path = Path(filename) 
    state: dict[str, Parameter] = module.state_dict()
    metadata: list[Metadata] = []  

    with open(f'{path.stem}.tannic', 'wb') as file:   
        nbytes = sum(parameter.nbytes for parameter in module.parameters()) 
        header = Header(magic=MAGIC, version=1, checksum=0xABCD, nbytes=nbytes) 
        offset = 0
        file.write(header.pack())  
        for name, parameter in state.items():  
            metadata.append(Metadata( 
                dcode=dcodeof(parameter.dtype),
                offset=offset,
                nbytes=parameter.nbytes,   
                namelength=len(name),
                name=name,  
            )) 
            offset += parameter.nbytes   
            file.write(parameter.detach().cpu().numpy().tobytes())   

    with open(f'{path.stem}.metadata.tannic', 'wb') as file:     
        nbytes = sum(calcsize(obj.format) + obj.namelength for obj in metadata)
        header = Header(magic=MAGIC, version=1, checksum=0xABCD, nbytes=nbytes) 
        file.write(header.pack())  
        print(metadata)
        for object in metadata: 
            file.write(object.pack())       
            file.write(object.name.encode('utf-8'))