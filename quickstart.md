---
title:
layout: default
---

Quick Start Guide
=========================================

## Prerequisites

This QuickStart guide assumes that you have the following already installed:

1. Java
2. An Integrated Development Environment (IDE) like IntelliJ
3. [Maven](../maven.html) (Java's automated build tool)
4. [Canova](../canova.html) (An ML Vectorization lib)
5. Github (Optional)
 
If you need to install any of the above, please read how in this [Getting Started guide](http://nd4j.org/getstarted.html).

## DL4J in 5 Easy Steps

After those installs, if you can follow these five steps, you'll be up and running:

1. *git clone* [the examples](https://github.com/deeplearning4j/dl4j-0.0.3.3-examples). We are currently on version 0.0.3.3.x.
2. Import the examples as a project into IntelliJ with Maven
3. Pick a Blas [backend](http://nd4j.org/dependencies.html) and insert it in your POM (Probably *nd4j-jblas*)

     <dependency>
       <groupId>org.nd4j</groupId>
       <artifactId>nd4j-$BACKEND_OF_YOUR_CHOICE</artifactId>
       <version>${nd4j.version}</version>
     </dependency>

4. Select example from the lefthand file tree (Start with *DBNSmallMnistExample.java*)
5. Hit run! (It's the green button)

Once you do that, try the other examples and see what they look like. 

## Dependencies in Maven

When you know which backend you want, search for it on [Maven Central](https://search.maven.org); click the linked version number under "Latest Version"; copy the dependency code on the left side of the subsequent screen; and paste it into your project root's pom.xml in IntelliJ.

For core algorithms, you can simply add this snippet to your deeplearning4j POM.xml file:

     <dependency>
         <groupId>org.deeplearning4j</groupId>
         <artifactId>deeplearning4j-core</artifactId>
         <version>${deeplearning4j.version}</version>
     </dependency>
     
For Natural-Language Processing (NLP):

     <dependency>
         <groupId>org.deeplearning4j</groupId>
         <artifactId>deeplearning4j-nlp</artifactId>
         <version>${deeplearning4j.version}</version>
     </dependency>

For Scaleout (Hadoop/Spark):

### Hadoop

      <dependency>
          <groupId>org.deeplearning4j</groupId>
          <artifactId>cdh4</artifactId>
          <version>${deeplearning4j.version}</version>
      </dependency>

### Spark

      <dependency>
          <groupId>org.deeplearning4j</groupId>
          <artifactId>spark</artifactId>
          <version>${deeplearning4j.version}</version>
      </dependency>

Installing From Source 
==============================

YOU DON'T HAVE TO DO THIS IF YOU'RE JUST USING THE SOFTWARE FROM MAVEN CENTRAL OR THE DOWNLOADS.

1. Download [Maven](http://maven.apache.org/download.cgi) and [set it up in your path](http://architectryan.com/2012/10/02/add-to-the-path-on-mac-os-x-mountain-lion/#.VVkVM9pVikp).
2. Run setup.sh on Unix or setup.bat on Windows. (The .sh and .bat files are in the root of the Deeplearning4j [git repo](https://github.com/deeplearning4j/deeplearning4j). To download that, set up [Github](http://nd4j.org/getstarted.html#github) and do a *git clone*. If you've already cloned the repo, do a *git pull*.)
