Jumpy: Python interface for [nd4j](https://nd4j.org)
===========================================

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/deeplearning4j/deeplearning4j/blob/master/jumpy/LICENSE)
[![PyPI version](https://badge.fury.io/py/jumpy.svg)](https://badge.fury.io/py/jumpy)

Jumpy allows you to use ND4J from Python _without any network communication_. Many other Python libraries bridging Java
have considerable overhead, jumpy uses pointers to directly access your numpy arrays. Under the hood, Jumpy uses `pydl4j` 
for dependency management and `pyjnius` to load Java classes.

## Installation

Jumpy is on PyPI, simply install it with

```bash
pip install jumpy
```

or build it from source:

```bash
python setup.py install
```

## Using Jumpy

### Creating arrays

Just like numpy, you can initialize an array using `.zeros()` or `.ones()`

```python
import jumpy as jp

x = jp.zeros((32, 16))
y = jp.ones((32, 16))
```

### Converting numpy array to jumpy array

A numpy `ndarray` instance can be converted to a jumpy `ndarray` instance (and vice-versa) without copying the data

```python
import jumpy as jp
import numpy as np

x_np = np.random.random((100, 50))
x_jp = jp.array(x_np)
```

### Converting jumpy array to numpy array

Simply call the `.numpy()` method of `jumpy.ndarray.ndarray`

```python
import jumpy as jp

x_jp = jp.zeros((100,50))
x_np = x_jp.numpy()
```

### Operations

* Basic operators like `+` `-` `*` `/` `+=` `-=` `*=` `/=` are overloaded and broadcasting is supported.
* Indexing, slicing and assignment behaviour has been made as close to numpy as possible.
* Check `jumpy/ops/` to see available ops.

---
## Contribute

* Check for open issues, or open a new issue to start a discussion around a feature idea or a bug.
* We could use more ops! Have a look at available ops (`jumpy/ops/`), it's quite easy to add new ones.
* Send a pull request and bug us on Gitter until it gets merged and published. :)
