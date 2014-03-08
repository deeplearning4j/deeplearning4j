---
title: 
layout: default
---

# data sets and machine learning

Machine learning typically works with three data sets: training, dev and test. All three should randomly sample a larger body of data.

The first set you use is the training set, the largest of the three. Running a training set through a neural network teaches the net how to weigh different features, assigning them coefficients according to their likelihood of minimizing errors in your results.

Those coefficients, also known as metadata, will be contained in vectors, one for each each layer of your net. They are one of the most important results you will obtain from training a neural network.

The second set, known as dev, is what you optimize against. While the first encounter between your algorithm and the data was unsupervised, the second requires your intervention. This is where you turn the knobs, adjusting coefficients to see which changes will help your network best recognize patterns by focusing on the crucial features.

The third set is your test. It functions as a seal of approval, and you don’t use it until the end. After you’ve trained and optimized your data, you test your neural net against this final random sampling. The results it produces should validate that your net accurately recognizes images, or at least [x] percentage of them.

If you don’t achieve validation, go back to dev, examine the quality of your data and look at your pre-processing techniques. If they do, that’s validation you can publish.





#Custom Data  Sets


To create a custom data set you need to create your own custom [../doc/org/deeplearning4j/datasets/iterator/DataSetIterator.html](DataSetIterator).

A  [../doc/org/deeplearning4j/datasets/iterator/DataSetIterator.html](DataSetIterator) knows how to iterate through a dataset and load it in to a

[../doc/org/deeplearning4j/datasets/DataSet.html](DataSet).


To help with this, there is a [../doc/org/deeplearning4j/datasets/iterator/BaseDataSetIterator.html](BaseDataSetIterator)

This is a class you can extend that offers basic functionality.

A [../doc/org/deeplearning4j/datasets/iterator/BaseDataSetIterator.html](BaseDataSetIterator) takes in something called a 


 [../doc/org/deeplearning4j/datasets/DataSetFetcher.html](DataSetFetcher) . A DataSetFetcher knows how to load data from images or text.

 It has a very similar API to the data set iterator, but is a bit lower level.