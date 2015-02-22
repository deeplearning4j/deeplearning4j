---
layout: default
---

# What is Deeplearning4j?

Deeplearning4j is the first commercial-grade, open-source, distributed deep-learning library written for Java and Scala. Integrated with Hadoop and Spark, DL4J is designed to be used in business environments, rather than as a research tool. It aims to be cutting-edge plug and play, more convention than configuration, which allows for fast prototyping for non-researchers.

### [Deep learning use cases](use_cases.html)

* Face/image recognition
* Voice search
* Speech-to-text (transcription)
* Spam filtering (anomaly detection)
* E-commerce fraud detection

### DL4J's main features

* A versatile [n-dimensional array](http://nd4j.org/) class. 
* [GPU](http://nd4j.org/gpu_native_backends.html) integration
* [Scalable](../scaleout.html) on [Hadoop](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/hadoop-yarn), [Spark](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-scaleout/spark) and Akka + AWS and other platforms

Deeplearning4j includes a distributed multi-threaded deep-learning framework and a normal single-threaded deep-learning framework. Training takes place in the cluster, which means it can process massive amounts of data. Nets are trained in parallel via iterative reduce, and they are equally compatible with **Java**, **Scala** and **Clojure**.

### DL4J's neural nets

* [Restricted Boltzmann machines](../restrictedboltzmannmachine.html)
* [Convolutional Nets](http://deeplearning4j.org/convolutionalnets.html) (images)
* [Stacked Denoising Autoencoders](../stackeddenoisingautoencoder.html) 
* Recurrent Nets/LSTMs (time series)
* Recursive autoencoders
* [Deep-belief networks](../deepbeliefnetwork.html)
* [Deep Autoencoders](http://deeplearning4j.org/deepautoencoder.html) (QA/information retrieval)
* [Recursive Neural Tensor Networks](http://deeplearning4j.org/recursiveneuraltensornetwork.html) (scenes, parsing)
* See our ["How to Choose a Neural Net" page](neuralnetworktable.html)

Deep neural nets are capable of [record-breaking accuracy](../accuracy.html). For a quick introduction to them, please visit our [overview](../overview.html) page. In a nutshell, Deeplearning4j lets you compose deep nets from various shallow nets, each of which form a layer. This flexibility lets you combine restricted Boltzmann machines, autoencoders, convolutional nets and recurrent nets as needed in a distributed, production-grade framework on Spark, Hadoop and elsewhere. 

There are a lot of parameters to adjust when you're training a deep-learning network. We've done our best to explain them, so that Deeplearning4j can serve as a DIY tool for Java, [Scala](https://github.com/deeplearning4j/nd4j/tree/master/nd4j-scala-api/src/main/scala/org/nd4j/api/linalg) and Clojure programmers.

For any question, please join [our Google Group](https://groups.google.com/forum/#!forum/deeplearning4j); for premium support, [contact us at Skymind](http://www.skymind.io/contact.html). [ND4J is the Java-based scientific computing engine](http://nd4j.org/) powering our matrix manipulations.

![Alt text](../img/logos_8.png)
