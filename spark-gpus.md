---
title: Running Deep Learning on Distributed GPUs With Spark
layout: default
---

# Running Deep Learning on Distributed GPUs With Spark

Deeplearning4j trains deep neural networks on distributed GPUs using Spark and CuDNN.

This post is a simple introduction to each of those technologies. It looks at each individually, and shows how Deeplearning4j pulls them together in an image processing example.

Spark was the Apache Foundation’s most popular project last year. As an open-source, distributed run-time, Spark can orchestrate multiple host threads. Deeplearning4j only relies on Spark as a data-access layer, since we have heavy computation needs that require more speed and capacity than Spark currently provides. It’s basically fast ETL.

CuDNN stands for the CUDA Deep Neural Network Library, and it was created by the GPU maker NVIDIA. CuDNN is one of the fastest libraries for deep convolutional networks. It ranks at or near the top of several [image-processing benchmarks](https://github.com/soumith/convnet-benchmarks) conducted by Soumith Chintala of Facebook. Deeplearning4j wraps CuDNN, and gives the Java community easy access to it. 

Deeplearning4j is the most widely used open-source deep learning tool for the JVM, including the Java, Scala and Clojure communities. Its aim is to bring deep learning to the production stack, integrating tightly with popular big data frameworks like Hadoop and Spark. DL4J works with all major data types – images, text, time series and sound – and includes algorithms such as convolutional nets, recurrent nets like LSTMs, NLP tools like word2vec and doc2vec, and various types of autoencoder.

Deeplearning4j is part of a free enterprise distribution called the Skymind Intelligence Layer, or SKIL. It is one of four open-source libraries maintained by Skymind. DL4J is powered by the scientific computing library ND4J, or n-dimensional arrays for Java, which performs the linear algebra and calculus necessary to train neural nets. ND4J is accelerated by a C++ library libnd4j. And finally, the DataVec library is used to vectorize all types of data.

Here’s an example of Deeplearning4j code that runs LeNet on Spark using GPUs.

First we configure Spark and load the data:

<script src="https://gist.github.com/agibsonccc/05887ab83055503cde7cdc2bf383689c.js"></script>

Then we configure the neural network:

<script src="http://gist-it.appspot.com/https://gist.github.com/agibsonccc/05887ab83055503cde7cdc2bf383689c?slice=63:114"></script>

Then we tell Spark how to perform parameter averaging:

<script src="http://gist-it.appspot.com/https://gist.github.com/agibsonccc/05887ab83055503cde7cdc2bf383689c?slice=114:122"></script>

And finally, we train the network by calling `.fit()` on `sparkNetwork`.

<script src="http://gist-it.appspot.com/https://gist.github.com/agibsonccc/05887ab83055503cde7cdc2bf383689c?slice=124:136"></script>
