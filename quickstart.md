---
title:
layout: default
---

Quick Start Guide
=========================================

## Prerequisites

This Quick Start guide assumes that you have the following already installed:

1. Java 7
2. IntelliJ (or another IDE)
3. Maven (Automated build tool)
4. Github 
 
If you need to install any of the above, please read how in the [ND4J Getting Started guide](http://nd4j.org/getstarted.html).  (ND4J is the linear-algebra engine powering Deeplearning4j, and the instructions there apply to both projects.) Don't install everything cited on that page, just the software cited above. 

We highly recommend you join our [Gitter Live Chat](gitter.im/deeplearning4j/deeplearning4j) if you have questions or feedback. Even some Java programmers are unfamiliar with Maven... If you're feeling anti-social or brashly independent, feel free to lurk and learn.

## DL4J in 5 Easy Steps

After those installs, if you can follow these five steps, you'll be up and running:

1. *git clone* [nd4j](https://github.com/deeplearning4j/nd4j/), [deeplearning4j](https://github.com/deeplearning4j/deeplearning4j/), [canova](https://github.com/deeplearning4j/Canova) and [the examples](https://github.com/deeplearning4j/dl4j-0.0.3.3-examples). We are currently on examples version 0.0.3.3.x.
2. From your console, run "mvn clean install -DskipTests -Dmaven.javadoc.skip=true" on *each* of those directories
3. Import the examples as a project into IntelliJ with Maven.
4. The default [backend](http://nd4j.org/dependencies.html) in the examples POM.xml is set to *nd4j-jblas* (For an easier install, Windows users should change that to *nd4j-java* in the dependencies.)
5. Select DBNIrisExample.java from the lefthand file tree
6. Hit run! (It's the green button)

You should get an F1 score about about 0.55, which is good for a small dataset like Iris. 

## Dependencies and Backends

Backends are what power the linear algebra operations behind DL4J's neural nets. Backends vary by chip. CPUs work fastest with both Jblas and Netlib Blas; GPUs with Jcublas. You can find all backends on [Maven Central](https://search.maven.org); click the linked version number under "Latest Version"; copy the dependency code on the left side of the subsequent screen; and paste it into your project root's pom.xml in IntelliJ. 

The *nd4j-java* backend will look something like this:

     <dependency>
       <groupId>org.nd4j</groupId>
       <artifactId>nd4j-java</artifactId>
       <version>${nd4j.version}</version>
     </dependency>

*nd4j-java* doesn't require Blas, which makes for the easiest setup on Windows. It works on all examples with DBNs, or deep-belief nets, but not on the other examples. 

The nd4j-jblas backend will look something like this:

     <dependency>
       <groupId>org.nd4j</groupId>
       <artifactId>nd4j-jblas</artifactId>
       <version>${nd4j.version}</version>
     </dependency>

*nd4j-jblas* works with all examples. To install Jblas, Windows users should refer to the [Deepelearining4j Getting Started page](../gettingstarted.html).

## Next Steps

Once you've run the examples, please visit our [Getting Started page](../gettingstarted.html) to explore further. And remember, DL4J is a multistep install. We highly recommend you join our [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) if you have questions or feedback, so we can walk you through it. If you're feeling anti-social or brashly independent, you're still invited to lurk and learn. 
