---
title: 
layout: default
---

*previous* - [custom datasets](../customdatasets.html)*
# loading data from csv's

It’s useful to know how to load data from CSV files into neural nets, especially when dealing with time series. There’s an easy way to do that with Deeplearning4j, because our DataSetIterator has a CSVDataSetIterator implementation. 

Here’s an explanation, in the comments, of the parameters necessary to instantiate the class:

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/iterator/CSVDataSetIterator.java?slice=13-25"></script>

And here’s an example of what a CSVDataSetIterator ought to look like (in the form of a unit test):

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/test/java/org/deeplearning4j/datasets/iterator/CSVDataSetIteratorTest.java?slice=10-16"></script>