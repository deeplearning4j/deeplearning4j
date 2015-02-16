---
title: 
layout: default
---

# DL4J vs. Torch vs. Pylearn/Theano

Deeplearning4j is not the first open-source deep-learning project, but it is distinguished from its predecessors in both programming language and intent. DL4J is a Java-based, industry-focused, **distributed deep-learning framework** intended to solve problems involving massive amounts of data in a reasonable amount of time. 

### Pylearn2/Theano

Most academic researchers in deep learning rely on [**Pylearn2**](http://deeplearning.net/software/pylearn2/) and [Theano](http://deeplearning.net/software/theano/), which are written in Python. Pylearn2 is a machine-learning library, while Theano is a library that handles multidimensional arrays, like Numpy. Both are powerful tools widely used for research purposes and serving a large community. They are well suited to data exploration and explicitly state that they are intended for research. 

Pylearn2 is a normal (non-distributed) framework that includes everything necessary to conduct experiments with multilayer Perceptrons, RBMs, Stacked Denoising Autoencoders and Convolutional nets. We recommend it for precisely those use cases. In contrast, Deeplearning4j intends to be the equivalent of Scikit-learn in the deep-learning space. It aims to automate as many knobs as possible. 

### Torch

[**Torch7**](http://torch.ch/) is a computational framework written in Lua that supports machine-learning algorithms. It is purported to be used by large tech companies that devote in-house teams to deep learning. Lua is a multi-paradigm language developed in Brazil in the early 1990s. 

Torch7, while powerful, [was not designed to be widely accessible](https://news.ycombinator.com/item?id=7929216) to the Python-based academic community, nor to corporate software engineers, whose lingua franca is Java. Deeplearning4j was written in Java to reflect our focus on industry and ease of use. We believe usability is the limiting parameter that inhibits more widespread deep-learning implementations. 

### Licensing

Licensing is another distinction: Both Theano and Torch employ a BSD License, which does not address patents or patent disputes. Deeplearning4j and ND4J are distributed under an **[Apache 2.0 License](http://en.swpat.org/wiki/Patent_clauses_in_software_licences#Apache_License_2.0)**, which contains both a patent grant and a litigation retaliation clause. That is, anyone is free to make and patent derivative works based on Apache 2.0-licensed code, but if they sue someone else over patent claims to the original code (DL4J in this case), they immediately lose all patent claim to it. (In other words, you are given resources to defend yourself in litigation, and discouraged from attacking others.)

### Speed

While both Torch7 and DL4J employ parallelism, DL4J's **parallelism is automatic**. That is, we automate the setting up of worker nodes and connections, allowing users to bypass libs while creating a massively parallel network. Deeplearning4j is best suited for solving specific problems, and doing so quickly. 

This brings us to the issue of speed. The bottleneck for speed that most deeplearning frameworks encounter is in [BLAS](http://www.netlib.org/blas/), or Basic Linear Algebra Subprograms, so if you use BLAS and the framework correctly, then you will [approach the same speed limit](https://www.quora.com/Deep-Learning/How-fast-is-Theano-compared-to-other-DBN-implementations). The question, however, is how easy a project's documentation makes it to handle large amounts of data.  

For a full list of Deeplearning4j's features, please see our [features page](../features.html).

### Why Java and Scala?

We're often asked why we chose to implement an open-source deep-learning project in Java, when so much of the deep-learning community is focused on Python. After all, Python has great syntactic elements that allow you to add matrices together without creating explicit classes, as Java requires you to do. Likewise, Python has an extensive scientific computing environment with native extensions like Theano and Numpy.

Yet Java has several advantages. First of all, as a language it is inherently faster than Python. Anything written in Python by itself, disregarding its reliance on Cython, will be slower. Admittedly, most computationally expensive operations are written in C or C++. (When we talk about operations, we also consider things like strings and other operations involved with higher-level machine learning processes.) Most deep-learning projects that are initially written in Python will have to be rewritten if they are to be put in production. Not so with Java.

Second, most major companies worldwide use Java or a Java-based system. It remains the most widely used language in the world. That is, many programmers solving real-world problems could benefit from deep learning, but they are separated from it by a language barrier. We want to make deep learning more usable to a large new audience that can put it to immediate use. 

Thirdly, Java's lack of robust scientific computing libraries can be solve by writing them, which we've done with [ND4J](http://nd4j.org), which runs on distributed GPUs or GPUs, and can be interfaced via a Java or Scala API.

###Ecosystem

Java's popularity is only strengthened by its ecosystem. Hadoop is implemented in Java; Spark runs within Hadoop's Yarn run-time; libraries like Akka made building distributed systems for Deeplearning4j feasible. In sum, Java boasts a highly tested infrastructure for pretty much any application, and deep-learning nets written in Java can live close to the data, which makes programmers' lives easier. Deeplearning4j can be run and provisioned as a YARN app.

Java can also be used natively from other popular languages like Scala, Clojure, Python and Ruby. By choosing Java, we excluded the fewest major programming communities possible. 

While Java is not as fast as C or C++, we've built a distributed system that can accelerate with the addition of more nodes. That is, if you want speed, just throw more boxes at it. 

Finally, we are building the basic applications of Numpy, including ND-Array, in Java for DL4J. Other features, such as GPU interoperability based on company-backed plugins, will be released shortly. We believe that many of Java's shortcomings can be solved quickly, and many of its advantages will continue for some time. 

### Scala

We have paid special attention to Scala in building Deeplearning4j and ND4J, because we believe Scala has the potential to become the language dominating data-science in the future. Writing numerical computing, vectorization and deep-learning libraries for the JVM moves the community toward that goal.

To really understand the differences between DL4J and other frameworks, you may just have to [try us out](http://deeplearning4j.org/quickstart.html).
