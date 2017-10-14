ND4J: Scientific Computing on the JVM
===========================================

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.nd4j/nd4j/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.nd4j/nd4j)
[![Javadoc](https://javadoc-emblem.rhcloud.com/doc/org.nd4j/nd4j/badge.svg)](http://nd4j.org/doc)

ND4J is an Apache 2.0-licensed scientific computing library for the JVM. By contributing code to this repository, you agree to make your contribution available under an Apache 2.0 license.

It is meant to be used in production environments rather than as a research tool, which means routines are designed to run fast with minimum RAM requirements.

Please search for the latest version on search.maven.org.

Or use the versions displayed in:
https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml


---
## Main Features

- Versatile n-dimensional array object
- Multiplatform functionality including GPUs
- Linear algebra and signal processing functions

Specifics

- Supports GPUs via with the CUDA backend nd4j-cuda-7.5 and Native via nd4j-native.
- All of this is wrapped in a unifying interface.
- The API mimics the semantics of Numpy, Matlab and scikit-learn.

---
## Modules
Several of these modules are different backend options for ND4J (including GPUs).

- api = core
- instrumentation
- jdbc = Java Database Connectivity
- jocl-parent = Java bindings for OpenCL
- scala-api = API for Scala users
- scala-notebook = Integration with Scala Notebook

---

## Building Specific Modules

It is possible to build the project without the native bindings. This can be done
by specic targeting of the project to build.

```
mvn clean package test -pl :nd4j-api
```

## Documentation

Documentation is available at [nd4j.org](http://nd4j.org/). Access the [JavaDocs](http://nd4j.org/doc/) for more detail.

---
## Installation

To install ND4J, there are a couple of approaches, and more information can be found on the [ND4J website](http://nd4j.org/getstarted.html).

#### Install from Maven Central

1. Search for nd4j in the [Maven Central Repository](http://mvnrepository.com/search?q=nd4j) to find the available nd4j jars.
2. Include the appropriate dependency in your pom.xml.

#### Clone from the GitHub Repo

https://deeplearning4j.org/buildinglocally 
## Contribute

1. Check for open issues, or open a new issue to start a discussion around a feature idea or a bug.
2. If you feel uncomfortable or uncertain about an issue or your changes, feel free to contact us on Gitter using the link above.
3. Fork [the repository](https://github.com/deeplearning4j/nd4j.git) on GitHub to start making your changes to the **master** branch (or branch off of it).
4. Write a test, which shows that the bug was fixed or that the feature works as expected.
5. Note the repository follows
   the [Google Java style](https://google.github.io/styleguide/javaguide.html)
   with two modifications: 120-char column wrap and 4-spaces indentation. You
   can format your code to this format by typing `mvn formatter:format` in the
   subproject you work on, by using the `contrib/formatter.xml` at the root of
   the repository to configure the Eclipse formatter, or by [using the INtellij
   plugin](https://github.com/HPI-Information-Systems/Metanome/wiki/Installing-the-google-styleguide-settings-in-intellij-and-eclipse).

6. Send a pull request, and bug us on Gitter until it gets merged and published.
