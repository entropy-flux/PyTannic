from struct import pack
from struct import calcsize
from pathlib import Path  
from torch.nn import Module
from torch.nn import Parameter
from pytannic.header import Header, MAGIC
from pytannic.torch.types import dcodeof
from pytannic.torch.parameters import Metadata

def write(module: Module, filename: str) -> None:
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