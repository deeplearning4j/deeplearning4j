---
title: Tensorflow & Deeplearning4j
layout: default
---

# Tensorflow & Deeplearning4j

Tensorflow and Deeplearning4j are complementary. Complimentary as in snacks on an airplane, and complementary as in they work together. Deeplearning4j has a model impport function. While our model import is chiefly focused on Tensorflow models created with Keras, later in 2017 it will apply directly to Tensorflow models. 

### <a name="tensorflow">TensorFlow</a>

* Google created TensorFlow to replace Theano. The two libraries are in fact quite similar. Some of the creators of Theano, such as Ian Goodfellow, went on to create Tensorflow at Google before leaving for OpenAI. 
* For the moment, **TensorFlow** does not support so-called “inline” matrix operations, but forces you to copy a matrix in order to perform an operation on it. Copying very large matrices is costly in every sense. TF takes 4x as long as the state of the art deep learning tools. Google says it’s working on the problem. 
* Like most deep-learning frameworks, TensorFlow is written with a Python API over a C/C++ engine that makes it run faster. Although there is experimental support for a Java API it is not currently considered stable, we do not consider this a solution for the Java and Scala communities. 
* TensorFlow runs dramatically [slower than other frameworks](https://arxiv.org/pdf/1608.07249v7.pdf) such as CNTK and MxNet. 
* TensorFlow is about more than deep learning. TensorFlow actually has tools to support reinforcement learning and other algos.
* Google's acknowledged goal with Tensorflow seems to be recruiting, making their researchers' code shareable, standardizing how software engineers approach deep learning, and creating an additional draw to Google Cloud services, on which TensorFlow is optimized. 
* TensorFlow is not commercially supported, and it’s unlikely that Google will go into the business of supporting open-source enterprise software. It's giving a new tool to researchers. 
* Like Theano, TensforFlow generates a computational graph (e.g. a series of matrix operations such as z = sigmoid(x) where x and z are matrices) and performs automatic differentiation. Automatic differentiation is important because you don't want to have to hand-code a new variation of backpropagation every time you're experimenting with a new arrangement of neural networks. In Google's ecosystem, the computational graph is then used by Google Brain for the heavy lifting, but Google hasn’t open-sourced those tools yet. TensorFlow is one half of Google's in-house DL solution. 
* From an enterprise perspective, the question some companies will need to answer is whether they want to depend upon Google for these tools. 
* Caveat: Not all operations in Tensorflow work as they do in Numpy. 

Pros and Cons

* (+) Python + Numpy
* (+) Computational graph abstraction, like Theano
* (+) Faster compile times than Theano
* (+) TensorBoard for visualization
* (+) Data and model parallelism
* (-) Slower than other frameworks
* (-) Much “fatter” than Torch; more magic
* (-) Not many pretrained models
* (-) Computational graph is pure Python, therefore slow
* (-) No commercial support
* (-) Drops out to Python to load each new training batch
* (-) Not very toolable
* (-) Dynamic typing is error-prone on large software projects

## TensorFlow's Java API

The TensorFlow web site acknowledges that "The TensorFlow Java API is not covered by the TensorFlow API stability guarantees."

For more information on using Java for deep learning, please see our [Quickstart page](https://deeplearning4j.org/quickstart). Join us in the fight [against the Borg](https://vimeo.com/84760450)! ;)

Deeplearning4j is faster than TensorFlow on [multi-GPUs](https://github.com/deeplearning4j/dl4j-benchmark). You can read about how to run your own [optimized DL4J benchmarks here](https://deeplearning4j.org/benchmark).

Deeplearning4j can import neural net models trained with Keras 1.0 on TensorFlow to run inference.
