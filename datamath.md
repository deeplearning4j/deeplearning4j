---
title: 
layout: default
---

# data's mathematical representations

Neural nets, and computers in general, only perceive the world as it is processed through mathematical transformations. Math is their seeing-eye dog. Unlike humans, they require data to be transformed into numeric representations before they can work with it. Deeplearning4j utilizes a number of built-in mathematical functions to manipulate and represent data; i.e. to make it workable. 

### normalization
[Normalization](../glossary.html#normalization) means adjusting values measured on different scales to a notionally common scale. Usually, it involves taking numbers of differing ranges and recalibrating them to a range between zero and one. 

### fourier transforms
[Fourier transforms](http://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/) are a form of signal processing. They are used with convolutional nets and can also be applied to time-series data. 

### tensors
[Tensors](https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/Tensor.java) are a way of representing data as n-dimensional matrices. They are the basis of recursive tensor networks, which are useful in identifying many different objects in a single scene or image. They make use of [convolutional networks](https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/util/Convolution.java), a more complicated and processing-intensive form of restricted Boltzmann machine.

### word2vec
We allow you to create word vectors with [Word2vec](../word2vec.html).

### pandas dataframe
[Pandas dataframe](http://pandas.pydata.org/pandas-docs/version/0.13.1/index.html) is a Python data analysis toolkit. The dataset class in DL4J implements a bare-bones version of the Pandas dataframe. 

* [MatrixUtil](https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/util/MatrixUtil.java) is where our matrix transformations live. 

* [Conjugate Gradient](https://github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/util/NonZeroStoppingConjugateGradient.java) is one of the optimization algorithms we have to train neural nets. 



