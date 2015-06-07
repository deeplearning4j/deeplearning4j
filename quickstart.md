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
3. Maven (Java's automated build tool)
4. Canova (An ML Vectorization lib)
5. Github (Optional)
 
If you need to install any of the above, please read how in this [Getting Started guide](http://nd4j.org/getstarted.html).

## DL4J in 5 Easy Steps

After those installs, if you can follow these five steps, you'll be up and running:

1) *git clone* [the examples](https://github.com/deeplearning4j/dl4j-0.0.3.3-examples)
2) Import the examples as a project into IntelliJ with Maven
3) Pick a Blas backend and insert it in your POM (You should probably choose *nd4j-jblas*)
4) Select example from the lefthand file tree (Start with DBNSmallMnistExample.java...)
5) Hit run! It's the green button.

Once you do that, try the other examples and see what they look like. 

Installing From Maven Central 
=========================================

You must install Maven first. ([See installation details here](http://nd4j.org/getstarted.html#maven) and read our [brief introduction to Maven here](../maven.html).)

Please use the latest version of the examples, which you will find in Maven. We are currently on 0.0.3.3.x.

Include an [ND4J](http://nd4j.org/) backend in this dependency in your deeplearning4j POM.xml file:

     <dependency>
       <groupId>org.nd4j</groupId>
       <artifactId>nd4j-$BACKEND_OF_YOUR_CHOICE</artifactId>
       <version>${nd4j.version}</version>
     </dependency>

These are the BLAS [backends](http://nd4j.org/gpu_native_backends.html) you can choose from:

     //Ensure that Cuda (for GPUs) is properly set up in your LD_LIBRARY_PATH
     
     nd4j-jcublas-${YOUR_CUDA_VERSION} 
     
     //Linux: Install Blas/Gfortran. OSX: Already set up. Windows: Set up the MINGW Blas libs on your path.
     
     nd4j-jblas 
     
     nd4j-netlib-blas
    
Versions can be found on [Maven Central](http://search.maven.org/#search%7Cga%7C2%7Cnd4j). When you know which backend you want, search for it there; click the linked version number under "Latest Version"; copy the dependency code on the left side of the subsequent screen; and paste it into your project root's pom.xml in IntelliJ.

For core algorithms, you can simply add this snippet to your deeplearning4j pom.xml file:

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
