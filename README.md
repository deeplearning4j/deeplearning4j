Jumpy: Python interface for [nd4j](https://nd4j.org)
===========================================

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Jumpy allows you to use nd4j from python via pointers (no network communication required like a lot of python libraries).

## Installation

Jumpy requires an uber jar (a jar file that contains nd4j and all its dependencies) and the path to this file is expected to be found in the environment variable `JUMPY_CLASS_PATH`.

Easiest way to build the uber jar is by running `mvn package` on the `pom.xml` file:

```bash 
git clone https://www.github.com/deeplearning4j/jumpy.git
cd jumpy
mvn package
```

This will create a jar file called `dl4j-1.0-SNAPSHOT.jar` in the `target` directory. Set `JUMPY_CLASS_PATH` environment variable to path of this file.

```bash
export JUMPY_CLASS_PATH='/...../jumpy/target/dl4j-1.0-SNAPSHOT.jar'
```

Finally, either install jumpy via pip:

```bash
pip install jumpy
```

Or from source:

```bash
python setup.py install
```
