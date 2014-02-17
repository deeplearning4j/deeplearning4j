---
title: 
layout: default
---

DataSets
====================================


[Datasets](../com/ccc/deeplearning/datasets/DataSet.html) are a pair of input output matrix pairs. Sub data sets ( one pair at a certain row)

are accessible via a get(i) similar to a list. The matrices of a data set are used as inputs and labels in to the neural networks in the

pretrain and finetune steps respectively.

Data sets are usually composed by [data set iterators](../com/ccc/deeplearning/datasets/iterator/DataSetIterator.html)

Data Set Iterators know how to compose and turn an input such as text or an image in to a matrix digestible

by a neural network.

Typical operations performed in data set iterators include data etl, transforms ([binarization](../glossary.html#binarization),[normalization](../glossary.html#normalization))
