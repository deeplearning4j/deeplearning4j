---
title: 
layout: default
---

# Deeplearning4j vs. Torch7 vs. Pylearn2

Deeplearning4j is not the first open-source deep-learning project, but it is distinguished from its predecessors in both programming language and intent. DL4J is a Java-based, industry-focused, **distributed deep-learning framework** intended to solve problems involving massive amounts of data in a reasonable amount of time. 

Most academic researchers in deep learning rely on [**Pylearn2**](http://deeplearning.net/software/pylearn2/) and [Theano](http://deeplearning.net/software/theano/), which are written in Python. Pylearn2 is a machine-learning library, while Theano is a library that handles multidimensional arrays, like Numpy. Both are powerful tools widely used for research purposes and serving a large community. They are well suited to data exploration and explicitly state that they are intended for research. 

Pylearn2 is a normal (non-distributed) framework that includes everything necessary to conduct experiments with multilayer Perceptrons, RBMs, Stacked Denoising Autoencoders and Convolutional nets. We recommend it for precisely those use cases. In contrast, Deeplearning4j intends to be the equivalent of Scikit-learn in the deep-learning space. It aims to automate as many knobs as possible. 

[**Torch7**](http://torch.ch/) is a computational framework written in Lua that supports machine-learning algorithms. It is purported to be used by large tech companies that devote in-house teams to deep learning. Lua is a multi-paradigm language developed in Brazil in the early 1990s. 

Torch7, while powerful, [was not designed to be widely accessible](https://news.ycombinator.com/item?id=7929216) to the Python-based academic community, nor to corporate software engineers, whose lingua franca is Java. Deeplearning4j was written in Java to reflect our focus on industry and ease of use. We believe usability is the limiting parameter that inhibits more widespread deep-learning implementations. 

While both Torch7 and DL4J employ parallelism, DL4J's **parallelism is automatic**. That is, we automate the setting up of worker nodes and connections, allowing users to bypass libs while creating a massively parallel network. Deeplearning4j is best suited for solving specific problems, and doing so quickly. 

For a full list of Deeplearning4j's features, please see our [features page](../features.html).

### Why Java?

We're often asked why we chose to implement an open-source deep-learning project in Java, when so much of the deep-learning community is focused on Python. After all, Python has great syntactic elements that allow you to add matrices together without creating explicit classes, as Java requires you to do. Likewise, Python has an extensive scientific computing environment with native extensions like Theano and Numpy.

Yet Java has several advantages. First of all, as a language it is inherently faster than Python. Anything written in Python by itself, disregarding its reliance on Cython, will be slower. Admittedly, most computationally expensive operations are written in C. (When we talk about operations, we also consider things like strings and other operations involved with higher-level machine learning processes.)

Second, most companies use Java or a Java-based system. It remains the most widely used language in the world. That is, many programmers solving real-world problems could benefit from deep learning, but they are separated from it by a language barrier. We want to make deep learning more usable to a large new audience that can put it to immediate use. 

Java's popularity is only strengthened by its ecosystem. Hadoop is implemented in Java; Spark runs within Hadoop's Yarn run-time; libraries like Akka made building distributed systems for Deeplearning4j feasible. In sum, Java boasts a highly tested infrastructure for pretty much any application. 

Java can also be used natively from other popular languages like Scala, Clojure, Python and Ruby. By choosing Java, we excluded the fewest major programming communities. 

While Java is not as fast as C or C++, we've built a distributed system that can accelerate with the addition of more nodes. That is, if you want speed, just throw more boxes at it. 

Finally, we are building the basic applications of Numpy, including ND-Array, in Java for DL4J. Other features, such as GPU interoperability based on company-backed plugins, will be released shortly. We believe that many of Java's shortcomings can be solved quickly, and many of its advantages will continue for some time. 
