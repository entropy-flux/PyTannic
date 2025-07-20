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