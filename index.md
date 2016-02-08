---
layout: default
title: Home
---

# What is Deeplearning4j?

Deeplearning4j is the first commercial-grade, open-source, distributed deep-learning library written for Java and Scala. Integrated with Hadoop and [Spark](../spark.html), DL4J is designed to be used in business environments, rather than as a research tool. [Skymind](http://skymind.io) is its commercial support arm.

Deeplearning4j aims to be cutting-edge plug and play, more convention than configuration, which allows for fast prototyping for non-researchers. DL4J is customizable at scale. Released under the Apache 2.0 license, all derivatives of DL4J belong to their authors.

By following the [instructions on our Quick Start page](../quickstart.html), you can run your first examples of trained neural nets in minutes.

### [Deep Learning Use Cases](../use_cases.html)

* [Face/image recognition](../facial-reconstruction-tutorial.html)
* Voice search
* Speech-to-text (transcription)
* Spam filtering (anomaly detection)
* E-commerce fraud detection
* [Regression](../linear-regression.html)

### Why Deeplearning4j? 

* A versatile [n-dimensional array](http://nd4j.org/) class for Java and Scala
* [GPU](http://nd4j.org/gpu_native_backends.html) integration
* [Scalable](../spark.html) on [Hadoop](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/hadoop-yarn), [Spark](../gpu_aws.html)
* [Canova](../canova.html): A General vectorization tool for Machine-Learning libs
* [ND4J: A linear algebra library 2x as fast as Numpy](http://nd4j.org/benchmarking)

Deeplearning4j includes both a distributed, multi-threaded deep-learning framework and a normal single-threaded deep-learning framework. Training takes place in the cluster, which means it can process massive amounts of data quickly. Nets are trained in parallel via [iterative reduce](../iterativereduce.html), and they are equally compatible with **Java**, **[Scala](http://nd4j.org/scala.html)** and **[Clojure](https://github.com/wildermuthn/d4lj-iris-example-clj/blob/master/src/dl4j_clj_example/core.clj)**. Deeplearning4j's role as a modular component in an open stack makes it the first deep-learning framework adapted for a [micro-service architecture](http://microservices.io/patterns/microservices.html).

### DL4J's Neural Networks

* [Restricted Boltzmann machines](../restrictedboltzmannmachine.html)
* [Convolutional Nets](../convolutionalnets.html) (images)
* [Recurrent Nets](../usingrnns.html)/[LSTMs](../lstm.html) (time series and sensor data)
* [Recursive autoencoders](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/layers/feedforward/autoencoder/recursive/RecursiveAutoEncoder.java)
* [Deep-belief networks](../deepbeliefnetwork.html)
* [Deep Autoencoders](http://deeplearning4j.org/deepautoencoder.html) (Question-Answer/data compression)
* Recursive Neural Tensor Networks (scenes, parsing)
* [Stacked Denoising Autoencoders](../stackeddenoisingautoencoder.html)
* For more, see ["How to Choose a Neural Net"](../neuralnetworktable.html)

Deep neural nets are capable of [record-breaking accuracy](../accuracy.html). For a quick neural net introduction, please visit our [overview](../neuralnet-overview.html) page. In a nutshell, Deeplearning4j lets you compose deep neural nets from various shallow nets, each of which form a so-called `layer`. This flexibility lets you combine restricted Boltzmann machines, other autoencoders, convolutional nets or recurrent nets as needed in a distributed, production-grade framework that works with Spark and Hadoop on top of distributed CPUs or GPUs.

Here's an overview of the different libraries we've built and where they fit into a larger system:

![Alt text](../img/schematic_overview.png)

There are a lot of parameters to adjust when you're training a deep-learning network. We've done our best to explain them, so that Deeplearning4j can serve as a DIY tool for Java, [Scala](https://github.com/deeplearning4j/nd4s) and [Clojure](https://github.com/whilo/clj-nd4j) programmers.

If you have any questions, please join [us on Gitter](https://gitter.im/deeplearning4j/deeplearning4j); for premium support, [contact us at Skymind](http://www.skymind.io/contact/). [ND4J is the Java-based scientific computing engine](http://nd4j.org/) powering our matrix operations. On large matrices, our benchmarks show [it runs roughly twice as fast as Numpy](http://nd4j.org/benchmarking).

### Deeplearning4j Tutorials

* [Using Recurrent Nets With DL4J](../usingrnns.html)
* [Deep-Belief Networks With MNIST](../mnist-tutorial.html)
* [IRIS Flower Dataset Tutorial](../iris-flower-dataset-tutorial.html)
* [Facial Reconstruction With Labeled Faces in the Wild](../facial-reconstruction-tutorial.html)
* [Customizing Data Pipelines With Canova](../image-data-pipeline.html)
* [Restricted Boltzmann machines](../restrictedboltzmannmachine.html)
* [Eigenvectors, PCA and Entropy](../eigenvector.html)
* [A Glossary of Deep-Learning Terms](../glossary.html)

### User Testimonial

      "I feel like Frankenstein. The doctor..." - Steve D. 
      
      "I am very keen on using deeplearning4j in production here. This is a massive opportunity in a market worth billions of quid." -John M.

### Contributing to Deeplearning4j

Developers who would like to contribute to Deeplearning4j can get started by reading our [Developer's Guide](../devguide.html).

### Research With Deeplearning4j

* Stanford NLP: "[Large-Scale Language Classification](http://nlp.stanford.edu/courses/cs224n/2015/reports/24.pdf)"
