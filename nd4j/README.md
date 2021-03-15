ND4J: Scientific Computing on the JVM
===========================================

[![Join the chat at https://gitter.im/deeplearning4j/deeplearning4j](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/deeplearning4j?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.nd4j/nd4j/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.nd4j/nd4j)
[![Javadoc](https://javadoc-emblem.rhcloud.com/doc/org.nd4j/nd4j/badge.svg)](https://deeplearning4j.org/api/latest/)

ND4J is an Apache 2.0-licensed scientific computing library for the JVM. By contributing code to this repository, you agree to make your contribution available under an Apache 2.0 license.

It is meant to be used in production environments rather than as a research tool, which means routines are designed to run fast with minimum RAM requirements.

Please search for the latest version on search.maven.org.

Or use the versions displayed in:
https://github.com/eclipse/deeplearning4j-examples/blob/master/pom.xml

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

## Documentation

Documentation is available at [deeplearning4j.org](https://deeplearning4j.org/). Access the [JavaDocs](https://deeplearning4j.org/api/latest/) for more detail.

---
## Installation

To install ND4J, there are a couple of approaches, and more information can be found on the [DL4J website](https://deeplearning4j.org/docs/latest/nd4j-overview).

#### Install from Maven Central

1. Search for nd4j in the [Maven Central Repository](https://search.maven.org/search?q=nd4j) to find the available nd4j jars.
2. Include the appropriate dependency in your pom.xml.

#### Clone from the GitHub Repo

https://deeplearning4j.org/docs/latest/deeplearning4j-build-from-source 

#### Build from sources

To build `ND4J` from sources launch from the present directory:

```shell script
$ mvn clean install -DskipTests=true
``` 

To run tests using CPU or CUDA backend run the following.

For CPU:

```shell script
$ mvn clean test -P testresources -P nd4j-testresources -P nd4j-tests-cpu -P nd4j-tf-cpu
```

For CUDA:

```shell script
$ mvn clean test -P testresources -P nd4j-testresources -P nd4j-tests-cuda -P nd4j-tf-gpu
```

## Contribute

1. Check for open issues, or open a new issue to start a discussion around a feature idea or a bug.
2. If you feel uncomfortable or uncertain about an issue or your changes, feel free to contact us on Gitter using the link above.
3. Fork [the repository](https://github.com/eclipse/deeplearning4j.git) on GitHub to start making your changes to the **master** branch (or branch off of it).
4. Write a test, which shows that the bug was fixed or that the feature works as expected.
5. Note the repository follows
   the [Google Java style](https://google.github.io/styleguide/javaguide.html)
   with two modifications: 120-char column wrap and 4-spaces indentation. You
   can format your code to this format by typing `mvn formatter:format` in the
   subproject you work on, by using the `contrib/formatter.xml` at the root of
   the repository to configure the Eclipse formatter, or by [using the IntelliJ
   plugin](https://github.com/HPI-Information-Systems/Metanome/wiki/Installing-the-google-styleguide-settings-in-intellij-and-eclipse).

6. Send a pull request, and bug us on Gitter until it gets merged and published.
