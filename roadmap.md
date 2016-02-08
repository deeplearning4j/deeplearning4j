---
title: Deeplearning4j Roadmap
layout: default
---

# Deeplearning4j Roadmap

These priorities have been set by what the Skymind has seen demand for among clients and open-source community members. Contributors are welcome to add features whose priority they deem to be higher. 

High priority:

* CUDA rewrite for ND4J (under way)
* Hyperparameter optimization (underway: [Arbiter](https://github.com/deeplearning4j/Arbiter))
* Parameter server
* Computation graph
* Sparse support for ND4J
* Performance tests for network training vs. other platforms (and where necessary: optimizations)
* Performance tests for Spark vs. local (ditto)
* Building examples at scale

Medium priority:

* OpenCL for ND4J
* CTC RNN (for speech etc.)

Nice to have:

* Automatic differentiation
* Proper complex number support for ND4J (+optimizations)
* Reinforcement learning
* Python support/interface
* Support for ensembles
* Variational autoencoders
* Generative adversarial models

Low priority:

* Hessian free optimization
* Other RNN types: multi-dimensional; attention models, Neural Turing Machine, etc.
* 3D CNNs

This is a work in progress. Last updated Jan. 14 2016.
