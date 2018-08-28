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

Run example:

```bash
cd examples
python basic.py
```

