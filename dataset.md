---
title: 
layout: default
---

Data sets
====================================

[Data sets](../org/deeplearning4j/datasets/DataSet.html) are input-output matrix pairs. 

Subdata sets (one pair at a certain row) are accessible via a get(i) similar to a list. The matrices of a data set are used as inputs and labels into the neural networks in the pretraining and finetuning steps, respectively.

Data sets are usually composed by [data set iterators](../com/ccc/deeplearning/datasets/iterator/DataSetIterator.html).

Data set iterators know how to compose and turn inputs such as text or image into a matrix digestible by a neural network.

Typical operations performed in data set iterators include data etl, transforms ([binarization](../glossary.html#binarization),[normalization](../glossary.html#normalization)).

Datasets are saveable as well. This can prevent you from having to recompute a data set of, say, word counts.

              //save
              DataSet d = ...;
              d.saveTo(new File("path"),isBinary);

              //load
              DataSet d = DataSet.load(new File("path"));



Operations on data
====================================

There are many kinds of transforms you can do on data which can lead to better accuracy of results.

2 of these include [Binarization](../glossary.html#binarization) and [Normalization](../glossary.html#normalization).

Binarization of data allows for marking the occurrence of an event while normalization allows for expressing the probability

of something happening. Both are more meaningful mathematically than just an arbitary number.