# PyDL4J - Java dependency management for Python applications

PyDL4J is a lightweight package manager for the DL4J ecosystem whick allows you to focus
on building Python applications on top of `pyjnius` without worrying about the details. You
can use PyDL4J for the following tasks:

- Automatically manage JARs for your Python projects, such as `jumpy` or `pydatavec`.
- Configure your Python DL4J environment through the PyDL4J command line interface.
- use PyDL4J as a replacement for Maven for basic tasks, from Python.

---------

[![Build Status](https://jenkins.ci.skymind.io/buildStatus/icon?job=deeplearing4j/pydl4j/master)](https: // jenkins.ci.skymind.io/blue/organizations/jenkins/deeplearing4j % 2Fpydl4j/activity)
[![License](https://img.shields.io/badge/License-Apache % 202.0-blue.svg)](https: // github.com/deeplearning4j/pydl4j/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/pydl4j.svg)](https: // badge.fury.io/py/pydl4j)

![PyDL4J](https: // github.com/deeplearning4j/pydl4j/blob/master/python_in_java.png)

# Installation

PyDL4J is on PyPI, so you can install it with `pip`:

```bash
pip install pydl4j
```

Alternatively, you can build the project locally as follows:

```bash
git clone https: // www.github.com/deeplearning4j/pydl4j.git
cd pydl4j
python setup.py install
```

As regular user, this will likely be enough for your needs. In fact, most of the time you
will not interact with PyDL4J directly at all. All other Python projects maintained by
Skymind use PyDL4J under the hood and will install this dependency for you.

# PyDL4J command line interface (CLI)

Installing PyDL4J exposes a command line tool called `pydl4j`. You can use this tool to configure
your PyDL4J environment. If you don't use the CLI, a default configuration that will be used instead.

**Note: ** If you intend to use the CLI, make sure to have[`docker` installed](https: // docs.docker.com/install/)
on your machine.

To initialize a new PyDL4j configuration, type

```bash
pydl4j init


██████╗ ██╗   ██╗██████╗ ██╗██╗  ██╗     ██╗
██╔══██╗╚██╗ ██╔╝██╔══██╗██║██║  ██║     ██║
██████╔╝ ╚████╔╝ ██║  ██║██║███████║     ██║
██╔═══╝   ╚██╔╝  ██║  ██║██║╚════██║██   ██║
██║        ██║   ██████╔╝███████╗██║╚█████╔╝
╚═╝        ╚═╝   ╚═════╝ ╚══════╝╚═╝ ╚════╝

pydl4j is a system to manage your DL4J dependencies from Python!

Which DL4J version do you want to use for your Python projects? (default '1.0.0-beta2'):
```

Follow the instructions provided by the CLI. At the end of this process you'll see a
JSON object carrying your configuration.

```bash
This is your current settings file config.json:

{
    "dl4j_core": true,
    "nd4j_backend": "cpu",
    "spark_version": "2",
    "datavec": false,
    "spark": true,
    "scala_version": "2.11",
    "dl4j_version": "1.0.0-beta2"
}

Does this look good? (default 'y')[y/n]:

```

If not configured otherwise, this configuration file will be stored at `~/.deeplearning4j/pydl4j/config.json`. This
configuration file is a lightweight version for Python users to avoid the cognitive load of the widely used
Project Object Model(POM) widely used in Java. PyDL4J will translate your configuration into the right format
internally to provide you with the tools you need.

Finally, to install the Java dependencies configured in your `config.json` you use the following command:

```bash
pydl4j install
```

This tool will install all necessary JARs into `~/.deeplearning4j/pydl4j` for you, by running `mvn` in a
docker container, and setting your classpath so that your `pyjnius` Python applications can access them.

# PyDL4J API

# Example

```python
import pydl4j
import jnius_config
from pydl4j import mvn

pydl4j.set_context('my_python_app_name')

# Fetch latest version of datavec.datavec-api from Maven central
pydl4j.mvn_install(group='datavec', artifact='datavec-api')

# Or fetch a specific version:
pydl4j.mvn_install(group='datavec', artifact='datavec-api',
                   version='1.0.0-beta')

jnius_config.set_classpath(pydl4j.get_dir())
```

# List all artifacts in a group

```python
mvn.get_artifacts(group_id)
```

# Example

```python
mvn.get_artifacts('datavec')
```

```bash
['datavec-api', 'datavec-arrow', 'datavec-camel', 'datavec-cli', 'datavec-data', 'datavec-data-audio', 'datavec-data-codec', 'datavec-d
 ata-image', 'datavec-data-nlp', 'datavec-dataframe', 'datavec-excel', 'datavec-geo', 'datavec-hadoop', 'datavec-jdbc', 'datavec-local',
 'datavec-nd4j-common', 'datavec-parent', 'datavec-perf', 'datavec-spark-inference-client', 'datavec-spark-inference-model', 'datavec-s
 park-inference-parent', 'datavec-spark-inference-server_2.10', 'datavec-spark-inference-server_2.11', 'datavec-spark_2.10', 'datavec-sp
 ark_2.11']
```

# List all versions of an artifact

```python
mvn.get_versions(group_id, artifact_id)
```

# Example

```python
mvn.get_versions('datavec', 'datavec-api')
```

```bash
['0.4.0', '0.5.0', '0.6.0', '0.7.0', '0.7.1', '0.7.2', '0.8.0',
    '0.9.0', '0.9.1', '1.0.0-alpha', '1.0.0-beta', '1.0.0-beta2']
```

# Get latest version of an artifact

```python
mvn.get_latest_version(group_id, artifact_id)
```

# Example

```python
mvn.get_latest_version('datavec', 'datavec-api')
```

```bash
'1.0.0-beta2'
```

# List all installed jars

```python
pydl4j.get_jars()
```

# Uninstall a jar

```python
# Find jar name from pydl4j.get_jars()
pydl4j.uninstall(jar_name)
```

# Uninstall all jars:

```python
pydl4j.clear_context()
```
