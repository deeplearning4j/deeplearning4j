---
title: Release Notes
layout: default
---

# Release Notes for Version 5.0

* FP16 support for CUDA
* [Better performance for multi-gpu}(http://deeplearning4j.org/gpu)
* Including optional P2P memory access support
* Normalization support for time series and images
* Normalization support for labels
* Removal of Canova and shift to DataVec: [Javadoc](http://deeplearning4j.org/datavecdoc/), [Github Repo](https://github.com/deeplearning4j/datavec)
* Numerous bug fixes
* Spark improvements

<p align="center">
<a href="http://deeplearning4j.org/quickstart" class="btn btn-custom" onClick="ga('send', 'event', ‘quickstart', 'click');">Get Started With Deeplearning4j</a>
</p>

## Release Notes for version 4.0

* Initial multi-GPU support viable for standalone and Spark. 
* Refactored the Spark API significantly
* Added CuDNN wrapper 
* Performance improvements for ND4J
* Introducing [DataVec](https://github.com/deeplearning4j/datavec): Lots of new functionality for transforming, preprocessing, cleaning data. (This replaces Canova)
* New DataSetIterators for feeding neural nets with existing data: ExistingDataSetIterator, Floats(Double)DataSetIterator, IteratorDataSetIterator
* New learning algorithms for word2vec and paravec: CBOW and PV-DM respectively
* New native ops for better performance: DropOut, DropOutInverted, CompareAndSet, ReplaceNaNs
* Shadow asynchronous datasets prefetch enabled by default for both MultiLayerNetwork and ComputationGraph
* Better memory handling with JVM GC and CUDA backend, resulting in significantly lower memory footprint

## Resources

* [Deeplearning4j on Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)
* [Deeplearning4j Source Code](https://github.com/deeplearning4j/deeplearning4j/)
* [ND4J Source Code](https://github.com/deeplearning4j/nd4j/)
* [libnd4j Source Code](https://github.com/deeplearning4j/libnd4j/)

## Roadmap for Summer 2016

* Deep Q Learning, integration with Open AI's Gym
* Deeplearning4s Scala API
* Standard NN configuration file shared with Keras
