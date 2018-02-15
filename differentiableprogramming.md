---
title: A Beginner's Guide to Differentiable Programming (DiffProg)
layout: default
---

# A Beginner's Guide to Differentiable Programming (DiffProg)

Yann LeCun described [differentiable programming](https://www.facebook.com/yann.lecun/posts/10155003011462143) like this:

```
Yeah, Differentiable Programming is little more than a rebranding of the modern collection Deep Learning techniques, the same way Deep Learning was a rebranding of the modern incarnations of neural nets with more than two layers.

The important point is that people are now building a new kind of software by assembling networks of parameterized functional blocks and by training them from examples using some form of gradient-based optimization….It’s really very much like a regular program, except it’s parameterized, automatically differentiated, and trainable/optimizable.

An increasingly large number of people are defining the networks procedurally in a data-dependent way (with loops and conditionals), allowing them to change dynamically as a function of the input data fed to them. It's really very much like a regular progam, except it's parameterized, automatically differentiated, and trainable/optimizable. Dynamic networks have become increasingly popular (particularly for NLP), thanks to deep learning frameworks that can handle them such as PyTorch and Chainer (note: our old deep learning framework Lush could handle a particular kind of dynamic nets called Graph Transformer Networks, back in 1994. It was needed for text recognition).

People are now actively working on compilers for imperative differentiable programming languages. This is a very exciting avenue for the development of learning-based AI.

Important note: this won't be sufficient to take us to "true" AI. Other concepts will be needed for that, such as what I used to call predictive learning and now decided to call Imputative Learning. More on this later....
```

Differentiable programming will never be the buzzword that deep learning is. It's too much of a mouthful. LeCun has proposed DP, DiffProg or dProg as less unwieldy substitutes.

### See Also

* [Software 2.0, by Andrej Karpathy](https://medium.com/@karpathy/software-2-0-a64152b37c35)

### <a name="beginner">Other Deep Learning Tutorials</a>

* [Introduction to Neural Networks](./neuralnet-overview)
* [Beginner's Guide to Reinforcement Learning](./deepreinforcementlearning)
* [Convolutional Networks (CNNs)](./convolutionalnets)
* [Generative Adversarial Networks (GANs)](./generative-adversarial-network)
* [Graph Data and Deep Learning](./graphdata)
* [Word2Vec: Neural Embeddings for NLP](./word2vec)
* [Symbolic Reasoning (Symbolic AI) & Deep Learning](./symbolicreasoning)
* [Restricted Boltzmann Machines](./restrictedboltzmannmachine)
* [Eigenvectors, Covariance, PCA and Entropy](./eigenvector)
* [Neural Networks & Regression](./logistic-regression)
* [Open Datasets for Machine Learning](./opendata)
* [Inference: Machine Learning Model Server](./modelserver)
