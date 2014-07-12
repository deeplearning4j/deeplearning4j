---
layout: default
---

# dataset methods

The class *DataSet* is made to hold a pair of input feature/label matrices. Labels are binarized: all labels considered true are 1s; the rest are 0s.

A DataSet can be created like this:

 <script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/DataSet.java?slice=60:65"></script>

The first parameter should specify the feature matrix, the second should point to the labels.

Once you have a DataSet, you'll want to know what to do with it. For neural nets to work well, DataSets require preprocessing, which standardizes data. Standardization means you'll be comparing apples to apples, and not some other strange fruit.

Two main ways to standardize data are through mean removal and variance scaling. Mean removal subtracts the mean of the data from every point in the set, effectively centering the set on zero. Variance scaling will "normalize" your data so that its maximum is 1 and its minimum is -1. This is achieved by dividing all data points by the standard deviation. Both mean removal and variance scaling can be accomplished with one method:

 <script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/DataSet.java?slice=226:234"></script>

Normalizing data is another way of saying you're putting it in the shape of a Bell curve. Sometimes you will want to inflate or deflate your DataSet to give it other shapes. Two methods that will do that are *multiplyBy* and *divideBy*. THIS IS GOOD FOR WHAT?

 <script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/DataSet.java?slice=137:144"></script>

You can also reshape a DataSet by defining the number of rows and columns you want it to include: THIS IS GOOD FOR WHAT?

 <script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/DataSet.java?slice=131:135"></script>

Binarization transforms continuous data into Boolean values by setting thresholds that will classify any data point below them as 0, and above them as 1.

 <script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/DataSet.java?slice=214:221"></script>

Finally, you'll want to know how to separate your test set from your training set. This is done with the method *splitTestAndTrain*:

 <script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/datasets/DataSet.java?slice=450:467"></script>