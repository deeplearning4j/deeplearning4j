---
title: 
layout: default
---

# Deeplearning4j Roadmap

These priorities have been set by what the Skymind has seen demand for among clients and open-source community members. Contributors are welcome to add features whose priority they deem to be higher. 

High priority:

* Parameter server
* Computation graph
* Sparse support for ND4J
* CUDA rewrite for ND4J (under way)
* Hyperparameter optimization (For now: random search. Bayesian methods next.)
* Performance tests for network training vs. other platforms (and where necessary: optimizations)
* Performance tests for Spark vs. local (ditto)

Medium priority:

* RNN variable length + one-to-many/many-to-one etc
* CTC RNN (for speech etc.)
* Early stopping (needed for hyperparameter optimization)
* Deepwalk/word2vec generalization/rewrite (see issue #930)

Nice to have:

* OpenCL for ND4J
* Automatic differentiation
* Proper complex number support for ND4J (+optimizations)
* Reinforcement learning
* Python support/interface
* Support for ensembles

Low priority:

* Hessian free
* Other RNN types; bidirectional, multi-dimensional; attention models, Neural Turing Machine, etc.
* 3D CNNs

This is a work in progress. Last updated Dec. 9 2015.
