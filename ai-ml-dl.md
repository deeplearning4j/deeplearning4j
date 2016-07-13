---
title: Artificial Intelligence, Machine Learning and Deep Learning
layout: default
---

# What's the difference between artificial intelligence, machine learning and deep learning?

Artificial intelligence is any computer program that does something smart. 

It can be a pile of if-then statements or a complex statistical model. Usually, when a computer program designed by AI researchers actually succeeds at something -- like winning at chess -- many people say it's "not really intelligent", because the algorithm's internals are well understood. So you could say that true AI is whatever computers can't do yet. ;)

Machine learning, as others here have said, is a subset of AI. That is, all machine learning counts as AI, but not all AI counts as machine learning. For example, symbolic logic (rules engines, expert systems and knowledge graphs) as well as evolutionary algorithms and Baysian statistics could all be described as AI, and none of them are machine learning. 

The "learning" part of machine learning means that ML algorithms attempt to optimize along a certain dimension; i.e. they usually try to minimize error or maximize the likelihood of their predictions being true. This has three names: an error function, a loss function, or an objective function, because the algorithm has an objective... When someone says they are working with a machine-learning algorithm, you can get to the gist of its value by asking: What's the objective function?

How does one minimize error? Well, one way is to build a framework that multiplies inputs in order to make guesses as to the inputs' nature. Different outputs/guesses are the product of the inputs and the algorithm. Usually, the initial guesses are quite wrong, and if you are lucky enough to have ground-truth labels pertaining to the input, you can measure how wrong your guesses are by contrasting them with the truth, and then use that error to modify your algorithm. That's what neural networks do. They keep on measuring the error and modifying their parameters until they can't achieve any less error. 

They are, in short, an optimization algorithm. If you tune them right, they minimize their error by guessing and guessing and guessing again.

