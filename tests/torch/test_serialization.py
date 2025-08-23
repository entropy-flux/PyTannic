import pytest
import torch
from pytannic.torch.tensors import serialize, deserialize
from pytannic.torch.types import dcodeof, dtypeof

@pytest.mark.parametrize("dtype", [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
])

@pytest.mark.parametrize("shape", [  
    (1,),              # 1D
    (2, 3),            # 2D
    (4, 2, 3),         # 3D
])
def test_serialize_deserialize_roundtrip(dtype, shape): 
    if dtype.is_complex:
        real = torch.randn(*shape, dtype=torch.float32)
        imag = torch.randn(*shape, dtype=torch.float32)
        x = real + 1j * imag
        x = x.to(dtype)
    else:
        x = torch.randn(*shape).to(dtype)
 
    if torch.cuda.is_available():
        x = x.to("cuda")

    serialized = serialize(x)
    x_restored = deserialize(serialized)
 
    assert isinstance(x_restored, torch.Tensor)
    assert x_restored.shape == x.shape
    assert x_restored.dtype == x.cpu().dtype
    if x.dtype.is_floating_point or x.dtype.is_complex:
        assert torch.allclose(x.cpu(), x_restored, atol=1e-6, rtol=1e-4)
    else:
        assert torch.equal(x.cpu(), x_restored)

def test_invalid_magic_header():
    with pytest.raises(Exception):
        bad_data = b"BADHDR" + bytes(100)
        deserialize(bad_data)

def test_non_tensor_input_raises():
    with pytest.raises(Exception):
        serialize("not a tensor") 