---
layout: default
---

# what is deeplearning4j?

Deeplearning4j is the first commercial-grade deep learning library written in java. It is meant to be used in business environments, rather than as a research tool for extensive data exploration.  Deeplearning4j has a dsl for neural networks however, that exposes all of the knobs. 

Deeplearning4j is most helpful in solving distinct problems (like identifying faces, voices, spam or e-commerce fraud). 

Deeplearning4j aims to be cutting-edge plug and play, more convention than configuration. By following its conventions, you get an infinitely scalable deep-learning architecture. 

Deeplearning4j is a **distributed deep learning framework** as well as a normal deep learning framework (you can run this on a single thread as well, and there are examples for both). Training takes place in the cluster, which means it can process massive amounts of data. Nets are trained in parallel via [iterative reduce](https://github.com/jpatanooga/KnittingBoar/wiki/Iterative-Reduce).

It's made for data input and neural net training at scale, and its output should be highly accurate predictive models. 

By following the links at the bottom of each page, you will learn to set up, and train with sample data, several types of deep-learning networks. These include single- and multithread networks, [Restricted Boltzmann machines](../restrictedboltzmannmachine.html), [deep-belief networks](../deepbeliefnetwork.html) and [Stacked Denoising Autoencoders](../stackeddenoisingautoencoder.html). 

For a quick introduction to neural nets, please see our [overview](../overview.html).

There are a lot of knobs to turn when you're training a deep-learning network. We've done our best to explain them, so that DL4J can serve as a DIY tool. If you would like premium support from the creators, please [contact us](http://www.skymind.io/contact.html), otherwise please feel free to use the google group.
