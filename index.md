---
layout: default
---

# What is Deeplearning4j?

Deeplearning4j is the first commercial-grade, __open-source__, distributed __deep-learning__ library written in __Java__. It is designed to be used in business environments, rather than as a research tool. DL4J aims to be cutting-edge plug and play, more convention than configuration, which allows for fast prototyping for non-researchers.

Here are a few problems you can solve with deep learning:

* [Face identification](../facial-reconstruction-tutorial.html)
* Image and voice search
* Speech to text (transcription)
* Spam filtering (anomaly detection)
* E-commerce fraud detection
* And [much more](use_cases.html)

Main features:

* A versatile [n-dimensional array](http://nd4j.org/) class. 
* [**GPU](http://nd4j.org/gpu_native_backends.html)** integration
* Totally [scalable](../scaleout.html) on **Hadoop**, Spark, AWS and other platforms

Deeplearning4j includes a distributed multi-threaded deep-learning framework and a normal single-threaded deep-learning framework. Training takes place in the cluster, which means it can process massive amounts of data. Nets are trained in parallel via iterative reduce, and they are equally compatible with **Java**, **Scala** and **Clojure**.

Here are some of the neural nets we support:

* [Restricted Boltzmann machines](../restrictedboltzmannmachine.html)
* [Deep-belief networks](../deepbeliefnetwork.html)
* [Deep Autoencoders](http://deeplearning4j.org/deepautoencoder.html)
* [Recursive Neural Tensor Networks](http://deeplearning4j.org/recursiveneuraltensornetwork.html)
* [Convolutional Nets](http://deeplearning4j.org/convolutionalnets.html)
* [Stacked Denoising Autoencoders](../stackeddenoisingautoencoder.html). 

For a quick introduction to neural nets, please visit our [overview](../overview.html) page. In a nutshell, Deeplearning4j lets you compose deep nets from various shallow nets, each of which form a layer. This flexibility allows you to combine restricted Boltzmann machines, autoencoders, convolutional nets and recurrent nets as you require -- all in a distributed, production-grade framework.

There are a lot of parameters to adjust when you're training a deep-learning network. We've done our best to explain them, so that Deeplearning4j can serve as a DIY tool for Java, Scala and Clojure programmers.

For any question, please join [our Google Group](https://groups.google.com/forum/#!forum/deeplearning4j); for premium support, [contact us at Skymind](http://www.skymind.io/contact.html). [ND4J is the Java scientific computing engine](http://nd4j.org/) powering our matrix manipulations.

![Alt text](../img/logos_8.png)
