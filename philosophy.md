---
title: DL4J GPU Support
layout: default
---

##Philosophy

#Research
Deeplearning4j is *not* a research framework. While it can be used for research, and we even encourage it (apache license),
we are very use case driven.
Most research is not directly useful for industry products. Many research papers being published (while having good ideas)
are barely usable until ideas mature.

One question we often get: "Why don't you have last week's paper implemented?" Because we spend time on tools 
that help people build products like [DataVec](http://deeplearning4j.org/DataVec) 

We recognize the need for flexibility, otherwise we wouldn't have the [Computation Graph](http://deeplearning4j.org/compgraph)
or [Keras import](http://deeplearning4j.org/keras). 

#Don't rewrite your data pipeline glue code
That is what [DataVec](http://deeplearning4j.org/DataVec)  is for.
Many people are in the habit of writing 1 off code for data pipelines.
If you are going to do that work anyways, why not contribute to 1 library?

#Spark
Another audience we get questions from is the spark community. They often ask why we're not more tightly integrated in to spark.
Pure and simple: Spark is a data access layer not a compute engine. Deep Learning needs things spark will likely never support
or have the interest in supporting the *right* way. The main thing there being hardware acceleration. We also work on mobile.
That being said, we do offer a [first class spark integration](http://deeplearning4j.org/spark) does exist. Many tools
will bolt on support with bash scripts by just talking to yarn. Dl4j is a spark job that knows how to access your gpu if it's present.

Our linear algebra library can detect the hardware the worker is running on and use whatever is there.
It allow allows fine grained control of the gpu and the compute environment via the CudaEnvironment singleton
when needed.

Another reason is the tooling for both binary and columnar. Dedicated data pipeline tools for machine learning are (especially on spark)
are academic in nature not targeted at real problems. With datavec and other common tools that we allow you to run
locally *and* on spark, you get the ability to use 1 data pipeline api across columnar data *and* binary (such as images and video)



#First class JVM

There is also this concept of first class jvm. Many frameworks that interact with spark do so via python.
When interacting with spark, you can do things like distributed grid search and other semi powerful things.

The problem here is python's speed and data bottleneck. [JNI is not slow](http://bytedeco.org/news/2015/03/14/java-meets-caffe/) 
thanks to javacpp though.

In finishing:
We are the data engineer and ops friendly framework. A jar file is something Central IT is ok deploying.
We will continue to build tools for business users.
People know how to use spark. People will continue to use other tools for research in python (which already has
the mind share and tooling) . What is missing from the ecosystem that we will continue to provide is a neutral (non cloud vendor lockin backed)
framework meant to run anywhere (including on prem)
