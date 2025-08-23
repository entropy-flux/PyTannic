from dataclasses import dataclass
from struct import pack, unpack, calcsize

@dataclass
class Metadata:
    dcode: int
    offset: int
    nbytes: int
    namelength: int  
    name: str

    FORMAT = "<B Q Q I"   
    def pack(self) -> bytes: 
        return pack(self.FORMAT, self.dcode, self.offset, self.nbytes, self.namelength)

    @classmethod
    def unpack(cls, data: bytes):
        fsize = calcsize(cls.FORMAT)
        dcode, offset, nbytes, namelength = unpack(cls.FORMAT, data[:fsize])
        return cls(dcode=dcode, offset=offset, nbytes=nbytes, namelength=namelength)

    @property
    def format(self) -> str:
        return self.FORMAT  