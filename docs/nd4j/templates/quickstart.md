---
title: Quickstart
short_title: Quick start tutorial
description: ND4J Key features and brief samples.
category: ND4J
weight: 1
---
<!--- Comments are standard html. Tripple dash based on stackoverflow: https://stackoverflow.com/questions/4823468/comments-in-markdown -->

<!--- Borrowing the layout of the Numpy quickstart to get started. -->

## Introduction
<!--- What is ND4J and why is it important. From the nd4j repo readme.  -->
ND4J is a scientific computing library for the JVM. It is meant to be used in production environments rather than as a research tool, which means routines are designed to run fast with minimum RAM requirements. The main features are:
* A versatile n-dimensional array object.
* Linear algebra and signal processing functions.
* Multiplatform functionality including GPUs.

This quickstart follows the same layout and approach of the [Numpy quickstart](https://docs.scipy.org/doc/numpy/user/quickstart.html). This should help people familiar with Python and Numpy get started quickly with Nd4J.

## Prerequisites
<!--- // Java, Maven, git. Coding skills and hello world example. -->

To follow the examples in this quick start you will need to know some Java. You will also need to install the following software on your computer:
<!--- from the dl4j quickstart, pointing to the dl4j quiclstart for details. -->
* [Java (developer version)](./deeplearning4j-quickstart#Java) 1.7 or later (Only 64-Bit versions supported)
* [Apache Maven](./deeplearning4j-quickstart#Maven) (automated build and dependency manager)
<!--- git allows us to start with a cleaner project than mvn create. -->
* [Git](./deeplearning4j-quickstart#Git)(distributed version control system)

If you are confident you know how to use maven and git, please feel free to skip to the [Basics](#Basics). In the remainder of this section we will build a small 'hello ND4J' application to verify the prequisites are set up correctly.

Execute the following commands to get the project from github. 

<!--- TODO: Create HelloNd4J or Quickstart-nd4j repo in Deeplearning4J. -->
```shell
git clone https://github.com/RobAltena/HelloNd4J.git

cd HelloNd4J

mvn install

mvn exec:java -Dexec.mainClass="HelloNd4j"
```

When everything is set up correctly you should see the following output:

```shell
SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".
SLF4J: Defaulting to no-operation (NOP) logger implementation
SLF4J: See http://www.slf4j.org/codes.html#StaticLoggerBinder for further details.
[         0,         0]
```

## Basics
<!--- TODO: We will put some into this page. Start with refering to existing doc. -->
While this quickstart is being build, please refer to our existing 
[basics usage](./nd4j-basics) document.

The main feature of Nd4j is the versatile n-dimensional array interface called INDArray. To improve performance Nd4j uses off-heap memory to store data. The INDArray is different from standard Java arrays.

Some of the key properties and methods for an INDArray x are as follows:

```java
// The number of axes (dimensions) of the array.
int dimensions = x.shape().length;

// The dimensions of the array. The size in each dimension.
long[] shape = x.shape();

// The total number of elements.
length = x.length();

// The type of the array elements. 
DataType dt = x.dataType();
```
<!--- staying away from itemsize and data buffer. The numpy quickstart has these. -->

### Array Creation
### Printing Arrays
### Basic Operations
### Universal Functions
### Indexing, Slicing and Iterating

## Shape Manipulation
### Changing the shape of an array
### Stacking together different arrays
### Splitting one array into several smaller ones

## Copies and View
### No Copy at All
### View or Shallow Copy
### Deep Copy

## Functions and Methods Overview
// List of links. Start with similar methods as the numpy quickstart.

// From here the Numpy quickstart goes deeper. For now we stop here.