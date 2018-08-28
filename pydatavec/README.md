# pydatavec : Python interface for DataVec

## Installation

Clone deeplearning4j mono repo:

```bash
git clone https://www.github.com/deeplearning4j/deeplearning4j.git
```

Switch to datavec directory:

```bash
cd deeplearning4j/datavec
```


Build datavec:

```bash
mvn clean install -D skipTests
```

Build datavec uberjar:
```bash
cd datavec-uberjar
mvn clean -P uberjar install -D skipTests
```

You will find the uberjar `datavec-uberjar-1.0.0-SNAPSHOT.jar` at `deeplearning4j/datavec/datavec-uberjar/target`

set `DATAVEC_CLASS_PATH` to the parent directory of this file.

This would look something like
```bash
export DATAVEC_CLASS_PATH=.../deeplearning4j/datavec/datavec-uberjar/target
```



Install pydatavec:

```bash
cd pydatavec
python setup.py install
```
## Examples

Examples are in the [dl4j-examples repo](www.github.com/deeplearning4j/dl4j-examples)

Clone dl4j-examples:

```bash
git clone https://www.github.com/deeplearning4j.dl4j-examples.git
```

Run examples in `pydatavec-examples` directory

```bash
cd pydatavec-examples
python basic.py
python iris.py
python reduction.py
```

