Jumpy: Python interface for [nd4j](https://nd4j.org)
===========================================

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Jumpy allows you to use nd4j from python via pointers (no network communication required like a lot of python libraries).

## Installation

Jumpy requires an uber jar (a jar file that contains nd4j and all its dependencies) and the path to this file is expected to be found in the environment variable `JUMPY_CLASS_PATH`.

Easiest way to build the uber jar is by running `mvn package` on the [`pom.xml`](/pom.xml) file:

```bash 
git clone https://github.com/deeplearning4j/deeplearning4j.git
cd jumpy
mvn package
```

This will create a jar file called `jumpy-1.0.0-beta.jar` in the `target` directory. Set `JUMPY_CLASS_PATH` environment variable to path of this file.

```bash
export JUMPY_CLASS_PATH='/...../jumpy/target/jumpy-1.0.0-SNAPSHOT.jar'
```

Finally, either install jumpy via pip:

```bash
pip install jumpy
```

Or from source:

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
