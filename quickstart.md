---
title: Quick Start Guide for Deeplearning4j
layout: default
---

Quick Start Guide
=================

Can't wait to start with Deeplearning4j? This is everything you need to run DL4J examples and begin your own projects.

We recommend that you join our [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j). You may want to lurk and learn at first - you can also request help and give feedback, but do use this guide before asking questions already answered below. If you are new to deep learning, we've included [a road map for beginners](../deeplearningforbeginners.html) with links to courses, readings and other resources. 

## Prerequisites

* [Java](#Java) 7 or later
* [Apache Maven](#Maven)
* [IntelliJ IDEA](#IntelliJ) (preferred IDE)
* [Git](#Git)

You *must* have all of these installed to use this Quick Start guide. DL4J targets professional Java developers who are familiar with production deployments, IDEs and automated build tools. Working with DL4J will be easiest if you already have experience in these areas.

If you are new to Java or unfamiliar with these tools, read the details below for help with installation and setup. **Otherwise, skip to** <a href="#examples">DL4J Examples</a>.

#### <a name="Java">Java</a>

If you don't have Java 7 or later, download the current [Java Development Kit (JDK) here](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html). To check if you have a compatible version of Java installed, use the following command:

```
java -version
```

#### <a name="Maven">Apache Maven</a>

Maven is a dependency management and automated build tool for Java projects. It works well with IDEs such as IntelliJ and lets you install DL4J project libraries easily. [Install or update Maven](https://maven.apache.org/download.cgi) to the latest release following [their instructions](https://maven.apache.org/install.html) for your system. To check if you have the most recent version of Maven installed, enter the following:

```
mvn --version
```

Maven is widely used among Java developers and it's not optional for working with DL4J. If you come from a different background and Maven is new to you, check out [Apache's Maven overview](http://maven.apache.org/what-is-maven.html) and our [introduction to Maven for non-Java programmers](http://deeplearning4j.org/maven.html). 

#### <a name="IntelliJ">IntelliJ IDEA</a>

An Integrated Development Environment ([IDE](http://encyclopedia.thefreedictionary.com/integrated+development+environment)) allows you to work with our API and build neural nets with a few clicks. We strongly recommend using [IntelliJ](https://www.jetbrains.com/idea/download/), which communicates with Maven to handle dependencies. The [community edition of IntelliJ](https://www.jetbrains.com/idea/download/) is freely available. 

There are other popular IDEs such as [Eclipse](http://books.sonatype.com/m2eclipse-book/reference/creating-sect-importing-projects.html) and [Netbeans](http://wiki.netbeans.org/MavenBestPractices). IntelliJ is preferred, and using it will make finding help on [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) easier if you need it.

#### <a name="Git">Git</a>

Install the [latest version of Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). If you already have Git, you can update to the latest version using Git itself:

```
$ git clone git://git.kernel.org/pub/scm/git/git.git
```

## <a name="examples">DL4J Examples in a Few Easy Steps</a>

1. Use command line to enter the following:

```
git clone https://github.com/deeplearning4j/dl4j-0.4-examples.git
```

2. Open IntelliJ and choose Import Project. Then select the main 'dl4j-0.4-examples' directory. 
![choose directory](../img/Install_IntJ_1.png)

3. Choose 'Import project from external model' and ensure that Maven is selected. 
![import project](../img/Install_IntJ_2.png)

4. Continue through the wizard's options, then click Finish. Wait a moment for IntelliJ to download all the dependencies. You'll see the horizontal bar working on the lower right.

5. Pick an example from the file tree on the left.
![run example in IntelliJ](../img/Install_IntJ_3.png)
Right-click the file to run. 

## Using DL4J In Your Own Projects: A Minimal pom.xml File

To run DL4J in your own projects, we highly recommend using Apache Maven for Java users, or a tool such as SBT for Scala. The basic set of dependencies and their versions are shown below. This includes:

- deeplearning4j-core, which contains the neural network implementations
- nd4j-x86, the CPU version of the ND4J library that powers DL4J
- canova-api - Canova is our library vectorizing and loading data


``` xml
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
Additional links:

- [Deeplearning4j artifacts on Maven Central](http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)
- [ND4J artifacts on Maven Central](http://search.maven.org/#search%7Cga%7C1%7Cnd4j)
- [Canova artifacts on Maven Central](http://search.maven.org/#search%7Cga%7C1%7Ccanova)

## Next Steps

1. Join us on Gitter. We have three big community channels.
  * [DL4J Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) is the main channel for all things DL4J. Most people hang out here.
  * [Tunning Help](https://gitter.im/deeplearning4j/deeplearning4j/tuninghelp) is for people just getting started with neural networks. Beginners please visit us here!
  * [Early Adopters](https://gitter.im/deeplearning4j/deeplearning4j/earlyadopters) is for those who are helping us vet and improve the next release. WARNING: This is for more experienced folks. 
2. Read the [introduction to deep neural networks](http://deeplearning4j.org/#tutorials) or one of our detailed tutorials. 
3. Check out the more detailed [Comprehensive Setup Guide](http://deeplearning4j.org/gettingstarted).
4. Browse all [DL4J documentation](http://deeplearning4j.org/documentation).
