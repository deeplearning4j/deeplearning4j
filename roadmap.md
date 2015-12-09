---
title: 
layout: default
---

# Deeplearning4j Roadmap

These priorities have been set by what the Skymind has seen demand for among clients and open-source community members. Contributors are free to add features themselves if they deem their priority to be higher. 

High priority:

•	Parameter server
•	Computation graph
•	Sparse support for ND4J
•	CUDArewrite for ND4J (under way)
•	Hyperparameter optimization (For now: random search. Bayesian methods would come next.)
•	Performance tests for network training vs. other platforms (and where necessary: optimizations)
•	Performance tests for Spark vs. local (and where necessary: optimizations)
Medium priority:
•	RNN variable length + one-to-many/many-to-one etc
•	CTC RNN (for speech etc)
•	Early stopping (but need this for hyperparameter optimization)
•	Deepwalk/w2v generalization/rewrite (deeplearning4j/deeplearning4j#930; Raver is looking into this)
Nice to have:
•	OpenCL for ND4J
•	Automatic differentiation
•	Proper complex number support for ND4J (+optimizations)
•	Reinforcement learning
•	Python support/interface
•	Support for ensembles
Low priority:
•	Hessian free
•	Other RNN types; bidirectional, multi-dimensional; attention, neural neuring machine, etc
•	3D CNNs

This is a work in progress. Last updated Dec. 9 2015.
