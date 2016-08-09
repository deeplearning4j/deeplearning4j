---
title: Artificial Intelligence, Machine Learning and Deep Learning
layout: default
---

# Artificial intelligence, machine learning & deep learning

You can think of deep learning, machine learning and artificial intelligence as a set of Russian dolls nested within each other, beginning with the smallest and working out. Deep learning is a subset of machine learning, which is a subset of AI.

AI is any computer program that does something smart, broadly speaking.

It can be a pile of if-then statements or a complex statistical model. Usually, when a computer program designed by AI researchers actually succeeds at something -- like winning at chess -- many people say it's "not really intelligent", because the algorithm's internals are well understood. So you could say that true AI is whatever computers can't do yet. ;)

Machine learning, as others here have said, is a subset of AI. That is, all machine learning counts as AI, but not all AI counts as machine learning. For example, symbolic logic (rules engines, expert systems and knowledge graphs) as well as evolutionary algorithms and Baysian statistics could all be described as AI, and none of them are machine learning.

<p align="center">
<a href="http://deeplearning4j.org/quickstart" class="btn btn-custom" onClick="ga('send', 'event', ‘quickstart', 'click');">Get Started With Deeplearning4j</a>
</p>

The "learning" part of machine learning means that ML algorithms attempt to optimize along a certain dimension; i.e. they usually try to minimize error or maximize the likelihood of their predictions being true. This has three names: an error function, a loss function, or an objective function, because the algorithm has an objective... When someone says they are working with a machine-learning algorithm, you can get to the gist of its value by asking: What's the objective function?

How does one minimize error? Well, one way is to build a framework that multiplies inputs in order to make guesses as to the inputs' nature. Different outputs/guesses are the product of the inputs and the algorithm. Usually, the initial guesses are quite wrong, and if you are lucky enough to have ground-truth labels pertaining to the input, you can measure how wrong your guesses are by contrasting them with the truth, and then use that error to modify your algorithm. That's what neural networks do. They keep on measuring the error and modifying their parameters until they can't achieve any less error.

They are, in short, an optimization algorithm. If you tune them right, they minimize their error by guessing and guessing and guessing again.

Deep learning is a subset of machine learning. Deep artificial neural networks are a set of algorithms setting new records in accuracy for many important problems, such as image recognition, sound recognition, recommender systems, etc. Deep learning is part of DeepMind's notorious AlphaGo algorithm, which beat the former world champion Lee Sedol at Go in early 2016. A more complete explanation of neural works is [here](./neuralnet-overview).

Deep is a technical term. It refers to the number of layers in a neural network. A shallow network has one so-called *hidden layer*, and a deep network has more than one. Multiple hidden layers allow deep neural networks to learn features of the data in a hierarchy, because simple features (e.g. two pixels) recombine from one layer to the next, to form more complex features (e.g. a line).

### <a name="beginner">Other Deeplearning4j Tutorials</a>
* [Introduction to Neural Networks](./neuralnet-overview)
* [Word2Vec: Neural Embeddings for Java](./word2vec)
* [Restricted Boltzmann Machines](./restrictedboltzmannmachine)
* [Eigenvectors, Covariance, PCA and Entropy](./eigenvector)
* [LSTMs and Recurrent Networks](./lstm)
* [Neural Networks and Regression](./linear-regression)
* [Convolutional Networks](./convolutionalnets)
