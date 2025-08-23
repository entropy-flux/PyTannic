from dataclasses import dataclass
from struct import pack, unpack

MAGIC = (69 | (82 << 8) | (73 << 16) | (67 << 24)) 

@dataclass
class Header:
    FORMAT = "<I B H Q"   
    magic: int
    version: int
    checksum: int
    nbytes: int 

    def pack(self) -> bytes:
        """Pack the header fields into bytes (little-endian)."""
        return pack(
            self.FORMAT,
            self.magic,
            self.version,
            self.checksum,
            self.nbytes, 
        )

    @classmethod
    def unpack(cls, data: bytes):
        """Unpack bytes into a Header instance (little-endian)."""
        unpacked = unpack(cls.FORMAT, data)
        return cls(*unpacked)
      