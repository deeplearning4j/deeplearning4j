---
title: Quick Start Guide for Deeplearning4j
layout: default
---

Quick Start Guide
=========================================

This page is designed to get you up and running with DL4J. 

## Prerequisites

This Quick Start guide assumes that you have the following already installed:

1. Java 7 or later
2. IntelliJ (or another IDE)
3. [Maven](https://maven.apache.org/) (Dependency management and automated build tool)
4. [Git](https://git-scm.com/)
 
If you need to install any of the above, please read the [ND4J Getting Started guide](http://nd4j.org/getstarted.html). (ND4J is the scientific computing engine we use to make deep learning work, and instructions there apply to both projects.) For the examples, don't install everything listed on that page, just the software listed above. 

We recommend that you join our [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) if you have questions or feedback. Even if you're feeling anti-social, feel free to lurk and learn. In addition, if you are brand-new to deep learning, we've included [a road map of what to learn when you're starting out](../deeplearningforbeginners.html). 

Deeplearning4j is an open-source project targetting professional Java developers familiar with production deployments, an IDE such as IntelliJ and an automated build tool such as Maven. Our tool will serve you best if you have those tools under your belt already.

## DL4J Examples in a Few Easy Steps

To start running the DL4J examples, follow these instructions:

* Enter `git clone https://github.com/deeplearning4j/dl4j-0.4-examples.git` in your command line
* In IntelliJ, create a new project using Maven by going to `File/New/Project from Existing Sources` in the menu tree. Point to the root directory of the examples above, and that will open them in your IDE. 
![creating a new project in IntelliJ](../img/IntelliJ_New_Project.png)
* Wait for IntelliJ to download all the dependencies. (You’ll see the horizontal bar working on the lower right.)
* Select an example from the lefthand file tree.
* Hit run! (It's the green button that appears when you right-click on the source file...)


## Using DL4J In Your Own Projects: A Minimal pom.xml File

To run DL4J in your own projects, we highly recommend using Apache Maven for Java users, or a tool such as SBT for Scala. The basic set of dependencies and their versions are shown below. This includes:

- deeplearning4j-core, which contains the neural network implementations
- nd4j-x86, the CPU version of the ND4J library that powers DL4J
- canova-api - Canova is our library vectorizing and loading data


```
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>MyGroupID</groupId>
    <artifactId>MyArtifactId</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <nd4j.version>0.4-rc3.8</nd4j.version>
        <dl4j.version>0.4-rc3.8</dl4j.version>
        <canova.version>0.0.0.14</canova.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-core</artifactId>
            <version>${dl4j.version}</version>
        </dependency>

        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-x86</artifactId>
            <version>${nd4j.version}</version>
        </dependency>
        
        <dependency>
            <artifactId>canova-api</artifactId>
            <groupId>org.nd4j</groupId>
            <version>${canova.version}</version>
        </dependency>
    </dependencies>
</project>
```

Optional dependencies you might want include:

- `deeplearning4j-ui`: contains the browser-based user-interface ([details here](http://deeplearning4j.org/visualization))
- `deeplearning4j-nlp`: contains the [Word2Vec](http://deeplearning4j.org/word2vec) implementation
- `dl4j-spark`: contains code for distributed training of neural networks on Apache Spark
- `canova-nd4j-image`: contains code for loading images
- `canova-nd4j-codec`: contains code for loading video files

Additional links:

- [Deeplearning4j artifacts on Maven Central](http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)
- [ND4J artifacts on Maven Central](http://search.maven.org/#search%7Cga%7C1%7Cnd4j)
- [Canova artifacts on Maven Central](http://search.maven.org/#search%7Cga%7C1%7Ccanova)

## Next Steps

Once you are done running the examples, have a look at our [documentation](http://deeplearning4j.org/documentation), our [more comprehensive setup guide](http://deeplearning4j.org/gettingstarted), or drop by [Gitter](https://gitter.im/deeplearning4j/deeplearning4j). We have three big community channels on Gitter: 1) The [main channel](https://gitter.im/deeplearning4j/deeplearning4j) for anything and everything. Most people hang out here. 2) The [tuning help channel](https://gitter.im/deeplearning4j/deeplearning4j/tuninghelp) for people just getting started with neural networks. Beginners please visit us here! 3) The [early adopters channel](https://gitter.im/deeplearning4j/deeplearning4j/earlyadopters) who are helping us vet and improve the next release. WARNING: This is for more experienced folks. 
