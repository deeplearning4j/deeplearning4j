---
title: 
layout: default
---

Data sets
====================================

[Data sets](../com/ccc/deeplearning/datasets/DataSet.html) are a pair of input-output matrix pairs. 

Subdata sets (one pair at a certain row) are accessible via a get(i) similar to a list. The matrices of a data set are used as inputs and labels into the neural networks in the pretraining and finetuning steps, respectively.

Data sets are usually composed by [data set iterators](../com/ccc/deeplearning/datasets/iterator/DataSetIterator.html).

Data set iterators know how to compose and turn an input such as text or image into a matrix digestible by a neural network.

Typical operations performed in data set iterators include data etl, transforms ([binarization](../glossary.html#binarization),[normalization](../glossary.html#normalization)).