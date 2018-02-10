---
title: Deep Learning for Graph Data
layout: default
---

# Deep Learning for Graph Data

Below are a few papers discussing how neural nets can be applied to data in graphs. 

Graphs are data structures that can be ingested by various algorithms, notably neural nets, learning to perform tasks such as classification, clustering and regression. 

In a nutshell, here's one way to make them ingestable for the algorithms: 

```
Data (graph, words) -> Real number vector -> Deep neural network
```

## Graph Data

Applying neural networks to graph data can de difficult. Neural nets do well on vectors and tensors; data types like images (which have structure embedded in them via pixel proximity); and sequences such as text and time series (which display structure in one direction, forward in time). Graphs do not have those kinds of structure. They have no proper beginning and no end, and two nodes connected to each other are not necessarily "close". 

![Alt text](./img/graph-data-example.png)

Algorithms can “embed” each node of a graph into a real vector (similar to the embedding a [word](./word2vec)). The result will be vector representation of each node in the graph with some information preserved. Once you have the real number vector, you can feed it to the neural network.

(This page is a WIP.)

<p align="center">
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">GET STARTED WITH DEEP LEARNING</a>
</p>

* [Community Detection with Graph Neural Networks](https://arxiv.org/abs/1705.08415) (2017)
by Joan Bruna and Xiang Li

*We study data-driven methods for community detection in graphs. This estimation problem is typically formulated in terms of the spectrum of certain operators, as well as via posterior inference under certain probabilistic graphical models. Focusing on random graph families such as the Stochastic Block Model, recent research has unified these two approaches, and identified both statistical and computational signal-to-noise detection thresholds.* 

*We embed the resulting class of algorithms within a generic family of graph neural networks and show that they can reach those detection thresholds in a purely data-driven manner, without access to the underlying generative models and with no parameter assumptions. The resulting model is also tested on real datasets, requiring less computational steps and performing significantly better than rigid parametric models.*

* [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652) (2014)

[DeepWalk is implemented in Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j/blob/1f8af820c29cc5567a2c5eaa290f094c4d1492a7/deeplearning4j-graph/src/main/java/org/deeplearning4j/graph/models/deepwalk/DeepWalk.java).

* [Deep Neural Networks for Learning Graph Representations](https://pdfs.semanticscholar.org/1a37/f07606d60df365d74752857e8ce909f700b3.pdf) (2016)
by Shaosheng Cao, Wei Lu and Qiongkai Xu

*In this paper, we propose a novel model for learning graph representations, which generates a low-dimensional vector
representation for each vertex by capturing the graph structural information. Different from other previous research efforts,
we adopt a random surfing model to capture graph structural information directly, instead of using the samplingbased
method for generating linear sequences proposed by Perozzi et al. (2014). The advantages of our approach will
be illustrated from both theorical and empirical perspectives. We also give a new perspective for the matrix factorization
method proposed by Levy and Goldberg (2014), in which the pointwise mutual information (PMI) matrix is considered as
an analytical solution to the objective function of the skipgram model with negative sampling proposed by Mikolov et
al. (2013). Unlike their approach which involves the use of the SVD for finding the low-dimensitonal projections from
the PMI matrix, however, the stacked denoising autoencoder is introduced in our model to extract complex features and
model non-linearities. To demonstrate the effectiveness of our model, we conduct experiments on clustering and visualization
tasks, employing the learned vertex representations as features. Empirical results on datasets of varying sizes show
that our model outperforms other state-of-the-art models in such tasks.*

* [Deep Feature Learning for Graphs](https://arxiv.org/abs/1704.08829)
by Ryan A. Rossi, Rong Zhou, Nesreen K. Ahmed

*This paper presents a general graph representation learning framework called DeepGL for learning deep node and edge representations from large (attributed) graphs. In particular, DeepGL begins by deriving a set of base features (e.g., graphlet features) and automatically learns a multi-layered hierarchical graph representation where each successive layer leverages the output from the previous layer to learn features of a higher-order. Contrary to previous work, DeepGL learns relational functions (each representing a feature) that generalize across-networks and therefore useful for graph-based transfer learning tasks. Moreover, DeepGL naturally supports attributed graphs, learns interpretable features, and is space-efficient (by learning sparse feature vectors). In addition, DeepGL is expressive, flexible with many interchangeable components, efficient with a time complexity of (|E|), and scalable for large networks via an efficient parallel implementation. Compared with the state-of-the-art method, DeepGL is (1) effective for across-network transfer learning tasks and attributed graph representation learning, (2) space-efficient requiring up to 6x less memory, (3) fast with up to 182x speedup in runtime performance, and (4) accurate with an average improvement of 20% or more on many learning tasks.*

* [Learning multi-faceted representations of individuals from heterogeneous evidence using neural networks](https://arxiv.org/abs/1510.05198) (al et Jurafsky)

* [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) (Stanford, 2016)

* [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) (Toronto and Microsoft, 2017)
by Yujia Li, Daniel Tarlow, Marc Brockschmidt and Richard Zemel

*Graph-structured data appears frequently in domains including chemistry, natural language semantics, social networks, and knowledge bases. In this work, we study feature learning techniques for graph-structured inputs. Our starting point is previous work on Graph Neural Networks (Scarselli et al., 2009), which we modify to use gated recurrent units and modern optimization techniques and then extend to output sequences. The result is a flexible and broadly useful class of neural network models that has favorable inductive biases relative to purely sequence-based models (e.g., LSTMs) when the problem is graph-structured. We demonstrate the capabilities on some simple AI (bAbI) and graph algorithm learning tasks. We then show it achieves state-of-the-art performance on a problem from program verification, in which subgraphs need to be matched to abstract data structures.*

## [Thought vectors](./thoughtvectors)

On a biological level, thoughts are literally graphs, graphs that capture the connections and action between neurons. Those graphs can represent a network of neurons whose connections fire in different ways over time as synapses fire, a dynamic flow of graphs. 
