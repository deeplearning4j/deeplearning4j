---
title: Learn Java
layout: default
---

# Learn Java Programming

Java is the world's most widely used programming language, and the language of Hadoop. Here are a few resources that will help you learn how to program in Java.

* [Learn Java The Hard Way](https://learnjavathehardway.org/)
* [Java Resources](http://wiht.link/java-resources)
* [Java Ranch: A Community for Java Beginners](http://javaranch.com/)
* [Intro to Programming in Java @Princeton](http://introcs.cs.princeton.edu/java/home/)
* [Head First Java](http://www.amazon.com/gp/product/0596009208)
* [Java in a Nutshell](http://www.amazon.com/gp/product/1449370829)

## Prerequisites

If you're going to do any work in Java at all, we recommend these tools.

* [Java (developer version)](#Java) 1.7 or later (**Only 64-Bit versions supported**)
* [Apache Maven](#Maven) (automated build and dependency manager)
* [IntelliJ IDEA](#IntelliJ) or Eclipse
* [Git](#Git) (version control system)

#### <a name="Java">Java</a>

If you don't have Java 1.7 or later, download the current [Java Development Kit (JDK) here](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html). To check if you have a compatible version of Java installed, use the following command:

``` shell
java -version
```

Please make sure you have a 64-Bit version of java installed, as you will see an error telling you `no jnind4j in java.library.path` if you decide to try to use a 32-Bit version instead.

#### <a name="Maven">Apache Maven</a>

Maven is a dependency management and automated build tool for Java projects. It works well with IDEs such as IntelliJ and lets you install DL4J project libraries easily. [Install or update Maven](https://maven.apache.org/download.cgi) to the latest release following [their instructions](https://maven.apache.org/install.html) for your system. To check if you have the most recent version of Maven installed, enter the following:

``` shell
mvn --version
```

If you are working on a Mac, you can simply enter the following into the command line:

``` shell
brew install maven
```

Maven is widely used among Java developers and it's pretty much mandatory for working with DL4J. If you come from a different background and Maven is new to you, check out [Apache's Maven overview](http://maven.apache.org/what-is-maven.html) and our [introduction to Maven for non-Java programmers](http://deeplearning4j.org/maven.html), which includes some additional troubleshooting tips. [Other build tools](../buildtools) such as Ivy and Gradle can also work, but we support Maven best.

* [Maven In Five Minutes](http://maven.apache.org/guides/getting-started/maven-in-five-minutes.html)

#### <a name="IntelliJ">IntelliJ IDEA</a>

An Integrated Development Environment ([IDE](http://encyclopedia.thefreedictionary.com/integrated+development+environment)) allows you to work with our API and configure neural networks in a few steps. We strongly recommend using [IntelliJ](https://www.jetbrains.com/idea/download/), which communicates with Maven to handle dependencies. The [community edition of IntelliJ](https://www.jetbrains.com/idea/download/) is free. 

There are other popular IDEs such as [Eclipse](http://books.sonatype.com/m2eclipse-book/reference/creating-sect-importing-projects.html) and [Netbeans](http://wiki.netbeans.org/MavenBestPractices). IntelliJ is preferred, and using it will make finding help on [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) easier if you need it.

#### <a name="Git">Git</a>

Install the [latest version of Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). If you already have Git, you can update to the latest version using Git itself:

``` shell
$ git clone git://git.kernel.org/pub/scm/git/git.git
```

Curious about deep learning in Java? Try starting here:

* [Introduction to Deep Neural Networks](./neuralnet-overview)
