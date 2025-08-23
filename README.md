# PyTannic
Python bindings for the Tannic framework. 

**PyTannic** is a Python library for interacting with the [Tannic-NN](https://github.com/entropy-flux/Tannic-NN) framework. It provides utilities for serializing PyTorch models, handling metadata, and sending/receiving tensor data over TCP in a format compatible with Tannic-NN.

---

## Features

- Serialize `torch.nn.Module` parameters.
- Network client for sending/receiving serialized tensors.

---

## Installation

```bash
pip install pytannic
```
## Quick Start

Write trained modules to files to load them from the C++ Tannic framework. 

```python
from pytannic.torch.modules import write
from torch.nn import Linear

module = Linear(10, 5)
write(module, "linear_model") 
```

PyTannic allows you to send serialized PyTorch tensors to a Tannic server and receive responses.

```python
tensor = torch.randn(3, 3)

with Client("127.0.0.1", 8080) as client:
    client.send(serialize(tensor)
    data = client.receive()
    result = deserialize(data)

print("Sent tensor:")
print(tensor)
print("\nReceived tensor:")
print(response)
```

This will allow you to easily create inference servers in C++. For a complete MNIST server example, see the [mnist-server repository](https://github.com/entropy-flux/mnist-server).

## License

PyTannic is licensed under the Apache License, Version 2.0. See the LICENSE file for details.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

