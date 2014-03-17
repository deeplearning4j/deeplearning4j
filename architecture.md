---
title: 
layout: default
---



Architecture
==============================




DeepLearning4j uses a host of technologies for both normal as well as distributed training and prediction of neural networks.

This is very likely to change and should be considered alpha as such. Below is the current architecture, and feedback is always appreciated.






Matrix Computations
=============================

[Jblas](http://mikiobraun.github.io/jblas/)


Jblas uses fortran underneath for matrix computations.

The reason for choosing jblas (and thus having native dependencies)

are the matrix computations being faster.


Neural network computaions are all computationally intense operations such as 

matrix multiples among other things, this was thus a requirement to be as fast as possible.





Distributed Service Discovery
======================================

[Zookeeper](http://zookeeper.apache.org/)

Zookeeper is great for distributed service discovery and configuration/cluster coordination.

It comes embedded in many environments already and powers a lot of famous technologies such as storm and hadoop.


Clustering / Distributed Computing
=============================================

[Akka](http://akka.io/)


Akka powers [Spark](http://spark.apache.org) underneath and is also a great parallel framework written in scala.

It has a great programming paradigm and has transparent parallel computation via the actor model.

This allows for at scale training, but very simple serializable models at prediction time.




Distributed Data Structures
=====================================

[Hazelcast](http://hazelcast.org/)

Hazelcast is a great distributed data structure framework for handling

common object serialization requirements. Hazelcast is primarily being used for transparent data structures for state tracking.

This may seem redundant with zookeeper, but zookeeper does not encourage/should not be used for distributed serialization stores.

Optimization Algorithms
==========================================

[Mallet](http://mallet.cs.umass.edu/optimization.php) is used for the optimization algorithms.

The default optimization algorithm used is conjugate gradient. There is also an implementation of

gradient descent in here as well. Descent and weight decrements are implemented via a reverse 

objective function. Since mallet is a maximizer rather than a minimzer, this will achieve

the intended affect wrt the objective functions and thus allow the use of maximization search algorithms.

