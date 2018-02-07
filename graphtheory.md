---
title: Graph Theory and Machine Learning
layout: default
---

# Graph Theory and Machine Learning

Here are a few papers discussing how neural nets can be applied to data encoded in graphs. (This page is a WIP.)

* [Community Detection with Graph Neural Networks](https://arxiv.org/abs/1705.08415) (2017)
by Joan Bruna and Xiang Li

*We study data-driven methods for community detection in graphs. This estimation problem is typically formulated in terms of the spectrum of certain operators, as well as via posterior inference under certain probabilistic graphical models. Focusing on random graph families such as the Stochastic Block Model, recent research has unified these two approaches, and identified both statistical and computational signal-to-noise detection thresholds. 
We embed the resulting class of algorithms within a generic family of graph neural networks and show that they can reach those detection thresholds in a purely data-driven manner, without access to the underlying generative models and with no parameter assumptions. The resulting model is also tested on real datasets, requiring less computational steps and performing significantly better than rigid parametric models.*

* [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652) (2014)

[DeepWalk is implemented in Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j/blob/1f8af820c29cc5567a2c5eaa290f094c4d1492a7/deeplearning4j-graph/src/main/java/org/deeplearning4j/graph/models/deepwalk/DeepWalk.java).

* [Learning multi-faceted representations of individuals from heterogeneous evidence using neural networks](https://arxiv.org/abs/1510.05198) (al et Jurafsky)

* [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) (Stanford 2016)

## [Thought vectors](./thoughtvectors)

On a biological level, thoughts are literally graphs, graphs that capture the connections and action between neurons. Those graphs can represent a network of neurons whose connections fire in different ways over time as synapses fire, a dynamic flow of graphs. 
