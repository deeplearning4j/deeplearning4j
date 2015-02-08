---
layout: default
---

# What is Deeplearning4j?

Deeplearning4j is the first commercial-grade, __open-source__, distributed deep-learning library written in Java. It is designed to be used in business environments, rather than as a research tool. DL4J aims to be cutting-edge plug and play, more convention than configuration, which allows for fast prototyping for non-researchers.

**Problems deep learning can solve**

* Face/image recognition
* Voice search
* Speech-to-text (transcription)
* Spam filtering (anomaly detection)
* E-commerce fraud detection
* And [other use cases](use_cases.html)

**DL4J's main features**

* A versatile [n-dimensional array](http://nd4j.org/) class. 
* [GPU](http://nd4j.org/gpu_native_backends.html) integration
* [Scalable](../scaleout.html) on Hadoop, Spark, Akka + AWS and other platforms

Deeplearning4j includes a distributed multi-threaded deep-learning framework and a normal single-threaded deep-learning framework. Training takes place in the cluster, which means it can process massive amounts of data. Nets are trained in parallel via iterative reduce, and they are equally compatible with **Java**, **Scala** and **Clojure**.

**DL4J's neural nets**

* [Restricted Boltzmann machines](../restrictedboltzmannmachine.html)
* [Convolutional Nets](http://deeplearning4j.org/convolutionalnets.html) (images)
* [Stacked Denoising Autoencoders](../stackeddenoisingautoencoder.html) 
* Recurrent Nets/LSTMs (time series)
* Recursive autoencoders
* [Deep-belief networks](../deepbeliefnetwork.html)
* [Deep Autoencoders](http://deeplearning4j.org/deepautoencoder.html) (QA/information retrieval)
* [Recursive Neural Tensor Networks](http://deeplearning4j.org/recursiveneuraltensornetwork.html) (scenes, parsing)

For a quick introduction to neural nets, please visit our [overview](../overview.html) page. In a nutshell, Deeplearning4j lets you compose deep nets from various shallow nets, each of which form a layer. This flexibility lets you combine restricted Boltzmann machines, autoencoders, convolutional nets and recurrent nets as needed in a distributed, production-grade framework. 

There are a lot of parameters to adjust when you're training a deep-learning network. We've done our best to explain them, so that Deeplearning4j can serve as a DIY tool for Java, Scala and Clojure programmers.

For any question, please join [our Google Group](https://groups.google.com/forum/#!forum/deeplearning4j); for premium support, [contact us at Skymind](http://www.skymind.io/contact.html). [ND4J is the Java-based scientific computing engine](http://nd4j.org/) powering our matrix manipulations.

![Alt text](../img/logos_8.png)
