from dataclasses import dataclass
from struct import calcsize, pack, unpack
from torch import Tensor
from torch import frombuffer
from pytannic.torch.types import dtypeof, dcodeof
from pytannic.header import Header, MAGIC 

@dataclass
class Metadata: 
    dcode: int
    offset: int
    nbytes: int
    rank: int
    shape: tuple[int, ...]  
 
    def pack(self) -> bytes:
        return pack(self.format, self.dcode, self.offset, self.nbytes, self.rank, *self.shape)

    @classmethod
    def unpack(cls, data: bytes):  
        head = calcsize("<B Q Q B")
        dcode, offset, nbytes, rank = unpack("<B Q Q B", data[:head])
        if rank == 0:
            shape = ()
        else:
            shape = unpack(f"<{rank}Q", data[head:head + rank * 8])
        return cls(dcode, offset, nbytes, rank, shape) 

    @property
    def format(self) -> str: 
        return f"<B Q Q B{self.rank}Q"
     

def serialize(tensor: Tensor) -> bytes:
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    rank = tensor.dim()
    shape  = tensor.shape
    buffer = tensor.numpy().tobytes() 
    metadata = Metadata(dcode=dcodeof(tensor.dtype), offset=0, nbytes=len(buffer), rank=rank, shape=shape)  
    header = Header(magic=MAGIC, version=1, checksum=0xABCD , nbytes = calcsize(metadata.format) + len(buffer)) 
    return header.pack() + metadata.pack()+ buffer 


def deserialize(data: bytes) -> Tensor:
    hsize = calcsize(Header.FORMAT) 
    head = calcsize("<B Q Q B")
    dcode, offset, nbytes, rank = unpack("<B Q Q B", data[hsize : hsize + head])
    
    msize = head + 8 * rank
    metadata = Metadata.unpack(data[hsize : hsize + msize]) 
    offset = hsize + msize
    buffer = bytearray(data[offset: offset + nbytes])
    return frombuffer(buffer, dtype=dtypeof(dcode)).reshape(metadata.shape) 