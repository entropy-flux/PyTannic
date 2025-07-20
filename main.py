'''
import struct
import torch
 
def dcodeof(dtype):
    if dtype == torch.int8:
        return 12
    elif dtype == torch.int16:
        return 13
    elif dtype == torch.int32:
        return 14
    elif dtype == torch.int64:
        return 15
    elif dtype == torch.float32:
        return 24
    elif dtype == torch.float64:
        return 25
    elif dtype == torch.complex64:
        return 37
    elif dtype == torch.complex128:
        return 38
    else:
        return 0  # none or unsupported
    
    import struct
from torch.nn import Module
from pathlib import Path

MAGIC = b'TANNIC'
import struct
from torch.nn import Module
from torch.nn import Parameter
from pathlib import Path
 

def write_module(model: Module, path: str):
    state_dict = model.state_dict()
    base_path = Path(path).stem
    data_path = f"{base_path}.tannic"
    meta_path = f"{base_path}.metadata.tannic"
 
    with open(data_path, "wb") as f_data:
        f_data.write(MAGIC)
        f_data.write(struct.pack("Q", len(state_dict)))  # Num tensors (for header)
        offsets = []
        
        for param in state_dict.values():
            tensor = param.detach().cpu().contiguous()
            data_bytes = tensor.numpy().tobytes()
            offsets.append(f_data.tell())  # Track buffer start
            f_data.write(struct.pack("Q", len(data_bytes)))  # Data size
            f_data.write(data_bytes)  # Raw data

    # 2. Write METADATA (names + everything else)
    with open(meta_path, "wb") as f_meta:
        f_meta.write(MAGIC)
        f_meta.write(struct.pack("Q", len(state_dict)))  # Num tensors

        for (name, param), offset in zip(state_dict.items(), offsets):
            tensor = param.detach().cpu().contiguous()
            name_bytes = name.encode('utf-8')
            
            # Pack metadata
            f_meta.write(struct.pack("Q", len(name_bytes)))  # Name length
            f_meta.write(name_bytes)  # Name
            f_meta.write(struct.pack("Q", offset))  # Offset in .tannic
            f_meta.write(struct.pack("B", tensor.ndim))  # Rank
            for dim in tensor.shape:  # Shape
                f_meta.write(struct.pack("Q", dim))
            for stride in tensor.stride():  # Strides
                f_meta.write(struct.pack("Q", stride))
            f_meta.write(struct.pack("B", dcodeof(tensor.dtype)))  # Dtype

    print(f"Exported:\n- Raw data: {data_path}\n- Metadata: {meta_path}")

write_module(model, 'model')
'''

from torch import dtype
from torch import (
    int8,
    int16,
    int32,
    int64,
    float32,
    float64,
    complex64,
    complex128,
)

def dcodeof(dtype: dtype):
    if dtype == int8:
        return 12
    elif dtype == int16:
        return 13
    elif dtype == int32:
        return 14
    elif dtype == int64:
        return 15
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

'''
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch

@dataclass
class TensorHeader:
    num_tensors: int  # Q (unsigned long long)

@dataclass
class TensorMetadata:
    name: str          # Q + utf-8 bytes
    offset: int        # Q
    ndim: int          # B (unsigned char)
    shape: Tuple[int]  # Q per dim
    stride: Tuple[int] # Q per stride
    dtype: torch.dtype # B (encoded)

@dataclass
class TensorData:
    byte_size: int    # Q
    bytes: bytes      # raw numpy bytes


def write_module(model: Module, path: str):
    state_dict = model.state_dict()
    base_path = Path(path).stem
    data_path = f"{base_path}.tnnc"
    meta_path = f"{base_path}.metadata.tnnc"

    # Prepare all data first
    tensor_data = []
    metadata = []
    current_offset = len(MAGIC) + 8  # MAGIC + header

    for name, param in state_dict.items():
        tensor = param.detach().cpu().contiguous()
        data_bytes = tensor.numpy().tobytes()
        
        # Store data
        tensor_data.append(TensorData(
            byte_size=len(data_bytes),
            bytes=data_bytes
        ))
        
        # Store metadata
        metadata.append(TensorMetadata(
            name=name,
            offset=current_offset,
            ndim=tensor.ndim,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype
        ))
        
        # Update offset (Q for byte_size + actual bytes)
        current_offset += 8 + len(data_bytes)

    # Write data file
    with open(data_path, "wb") as f_data:
        f_data.write(MAGIC)
        f_data.write(struct.pack("Q", len(tensor_data)))  # Header
        
        for data in tensor_data:
            f_data.write(struct.pack("Q", data.byte_size))
            f_data.write(data.bytes)

    # Write metadata file
    with open(meta_path, "wb") as f_meta:
        f_meta.write(MAGIC)
        f_meta.write(struct.pack("Q", len(metadata)))  # Header
        
        for meta in metadata:
            name_bytes = meta.name.encode('utf-8')
            f_meta.write(struct.pack("Q", len(name_bytes)))
            f_meta.write(name_bytes)
            f_meta.write(struct.pack("Q", meta.offset))
            f_meta.write(struct.pack("B", meta.ndim))
            for dim in meta.shape:
                f_meta.write(struct.pack("Q", dim))
            for stride in meta.stride:
                f_meta.write(struct.pack("Q", stride))
            f_meta.write(struct.pack("B", dcodeof(meta.dtype)))

    print(f"Exported:\n- Raw data: {data_path}\n- Metadata: {meta_path}")

''' 
from struct import pack
from pathlib import Path
from dataclasses import dataclass
from torch.nn import Module
from torch.nn import Parameter

@dataclass
class Metadata:  
    name:    str         
    dtype:   dtype 
    rank:    int       
    shape:   tuple[int]
    strides: tuple[int]
    offset:  int   
    nbytes:  int      

def write(module: Module, filename: str) -> None:
    path = Path(filename)
    state: dict[str, Parameter] = module.state_dict()
    metadata = list[Metadata]() 
    with open(f'{path.stem}.tnnc', 'wb') as file:
        file.write(b'TANNIC') 
        file.write(b'\x00' * (32 - 6))
        offset = 32
        for name, parameter in state.items(): 
            nbytes = parameter.nbytes
            metadata.append(Metadata(
                name=name, 
                dtype=parameter.dtype, 
                rank=parameter.dim(), 
                shape=tuple(parameter.size()), 
                strides=tuple(parameter.stride()),
                offset=offset,
                nbytes=nbytes
            ))
            offset += nbytes
            file.write(parameter.detach().cpu().numpy().tobytes())

    
    with open(f'{path.stem}.metadata.tnnc', 'wb') as file:
        file.write(b'TANNIC') 
        file.write(b'\x00' * (32 - 6))
        for object in metadata:
            file.write(pack("B", len(object.name)))           
            file.write(object.name.encode('utf-8'))
            file.write(pack("B", dcodeof(object.dtype))) 
            for dimension in range(object.rank):
                file.write(pack("Q", object.shape[dimension]))
                file.write(pack("Q", object.strides[dimension]))
            file.write(pack("Q", object.offset))
            file.write(pack("Q", object.nbytes))
