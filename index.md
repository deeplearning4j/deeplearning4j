---
layout: default
---

# What is Deeplearning4j?

Deeplearning4j is the first commercial-grade, open-source deep-learning library written in Java. It is meant to be used in business environments, rather than as a research tool for extensive data exploration. Deeplearning4j is most helpful in solving distinct problems, like identifying [faces](../facial-reconstruction-tutorial.html), voices, spam or e-commerce fraud. 

Deeplearning4j **integrates with GPUs** and includes a versatile **[n-dimensional array](http://nd4j.org/)** class. DL4J aims to be cutting-edge plug and play, more convention than configuration. By following its conventions, you get an [infinitely scalable](../scaleout.html) deep-learning architecture suitable for Hadoop and other big-data structures. This Java deep-learning library has a domain-specific language for neural networks that serves to turn their multiple knobs. 

Deeplearning4j includes a **distributed deep-learning framework** and a normal deep-learning framework (i.e. it runs on a single thread as well). Training takes place in the cluster, which means it can process massive amounts of data. Nets are trained in parallel via iterative reduce, and they are equally compatible with Java, Scala and Clojure, since they're written for the JVM. 

This open-source, distributed deep-learning framework is made for data input and neural net training at scale, and its output should be highly accurate predictive models. 

By following the links at the bottom of each page, you will learn to set up, and train with sample data, several types of deep-learning networks. These include single- and multithread networks, [Restricted Boltzmann machines](../restrictedboltzmannmachine.html), [deep-belief networks](../deepbeliefnetwork.html), [Deep Autoencoders](http://deeplearning4j.org/deepautoencoder.html), [Recursive Neural Tensor Networks](http://deeplearning4j.org/recursiveneuraltensornetwork.html), [Convolutional Nets](http://deeplearning4j.org/convolutionalnets.html) and [Stacked Denoising Autoencoders](../stackeddenoisingautoencoder.html). 

For a quick introduction to neural nets, please see our [overview](../overview.html).

There are a lot of knobs to turn when you're training a deep-learning network. We've done our best to explain them, so that Deeplearning4j can serve as a DIY tool for Java, Scala and Clojure programmers. If you have questions, please join [our Google Group](https://groups.google.com/forum/#!forum/deeplearning4j); for premium support, [contact us at Skymind](http://www.skymind.io/contact.html). The linear algebra engine powering our matrix manipulations is [ND4J](http://nd4j.org/).
