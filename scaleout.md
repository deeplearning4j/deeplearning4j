---
title: Scaleout
layout: default
---

# Scaleout

Deeplearning4j integrates with [Spark](http://deeplearning4j.org/gpu_aws.html), [Hadoop/YARN](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/hadoop-yarn) and can also spin out a stand-alone distributed system using Akka and AWS. 

It employs a method known as [iterative reduce](../iterativereduce) to process data in parallel. (Without some form of parallelism, training a neural network on large datasets can take weeks.)

Deeplearning4j can be called from within Hadoop as a YARN app. It is run and provisioned by Hadoop as a first-class citizen within the framework. Likewise, lines of Deeplearning4j code can be called from within Spark shell to initiate distributed neural net training. 
