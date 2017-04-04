---
title: Machine Learning With Deeplearning4j
layout: default
---

# Machine Learning With Deeplearning4j

While Deeplearning4j and its suite of open-source libraries - [ND4J](http://nd4j.org/), DataVec, Arbiter, etc. - primarily implement scalable, deep artificial neural networks, developers can also work with more traditional machine-learning algorithms using our framework.

## Algorithms Available on DL4J

* Linear Regression
* Logistic Regression
* [K-means clustering](https://deeplearning4j.org/doc/org/deeplearning4j/clustering/kmeans/package-tree.html )
* K nearest neighbor (k-NN)
* Optimizations of k-NN with a [VP-tree](https://en.wikipedia.org/wiki/Vantage-point_tree), [t-SNE](https://lvdmaaten.github.io/tsne/) and quad-trees as a side effect

## Algorithms Possible on ND4J

ND4J is a generic tensor library, so the sky's the limit on what can be implemented. 

We are integrating with Haifeng Li's SMILE, or [Statistical Machine Intelligence and Learning Engine](http://haifengl.github.io/smile/), which implements more than one hundred different statistical and machine-learning algorithms, including random forests and GBMs. SMILE shows the best performance of any open-source JVM-based machine-learning library we've seen. 
