---
layout: default
title: Home
---

# What is Deeplearning4j?

Deeplearning4j is the first commercial-grade, open-source, distributed deep-learning library written for Java and Scala. Integrated with Hadoop and [Spark](../spark.html), DL4J is designed to be used in business environments, rather than as a research tool. [Skymind](http://skymind.io) is its commercial support arm.

Deeplearning4j aims to be cutting-edge plug and play, more convention than configuration, which allows for fast prototyping for non-researchers. DL4J is customizable at scale. Released under the Apache 2.0 license, all derivatives of DL4J belong to their authors.

By following the [instructions on our Quick Start page](../quickstart.html), you can run your first examples of trained neural nets in minutes.

### [Deep Learning Use Cases](../use_cases)

* Face/image recognition
* Voice search
* Speech-to-text (transcription)
* Spam filtering (anomaly detection)
* [Fraud detection](http://www.skymind.io/finance/) 
* [Recommender Systems (CRM, adtech, churn prevention)](http://www.skymind.io/commerce/)
* [Regression](../linear-regression.html)

### Why Deeplearning4j? 

* A versatile [n-dimensional array](http://nd4j.org/) class for Java and Scala
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

### <a name="tutorials">Deeplearning4j Tutorials</a>

* [Introduction to Deep Neural Networks](../neuralnet-overview)
* [Convolutional Networks Tutorial](../convolutionalnets)
* [LSTM and Recurrent Network Tutorial](../lstm)
* [Using Recurrent Nets With DL4J](../usingrnns)
* [Deep-Belief Networks With MNIST](../mnist-tutorial)
* [Customizing Data Pipelines With Canova](../image-data-pipeline)
* [Restricted Boltzmann machines](../restrictedboltzmannmachine)
* [Eigenvectors, PCA and Entropy](../eigenvector)
* [A Glossary of Deep-Learning Terms](../glossary)

### User Testimonial

      "I feel like Frankenstein. The doctor..." - Steve D. 
      
      "I am very keen on using deeplearning4j in production here. This is a massive opportunity in a market worth billions of quid." -John M.

### Contributing to Deeplearning4j

Developers who would like to contribute to Deeplearning4j can get started by reading our [Developer's Guide](../devguide).

### Research With Deeplearning4j

* University of Massachusets "[RandomOut: Using a convolutional gradient norm to win The Filter Lottery](http://arxiv.org/abs/1602.05931)"
* Stanford NLP: "[Large-Scale Language Classification](http://nlp.stanford.edu/courses/cs224n/2015/reports/24.pdf)"
* [Like2Vec: A New Technique for Recommender Algorithms](https://docs.google.com/presentation/d/19QDuPmxB9RzQWKXp_t3yqxCvMBSMaOQk19KNZqUUgYQ/edit?pref=2&pli=1#slide=id.g11a4ba0c5c_0_6)
