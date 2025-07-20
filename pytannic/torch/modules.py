from struct import pack
from pathlib import Path
from dataclasses import dataclass
from torch import dtype
from torch.nn import Module
from torch.nn import Parameter
from pytannic.torch.types import dcodeof

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