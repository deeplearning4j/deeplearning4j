---
title: Deeplearning4j model saving
layout: default
---

# Model saving

In DL4J, you can save models via the [Persistable](../doc/org/deeplearning4j/nn/Persistable.html) interface.

Each network and dataset implement this interface for saving data. Which is useful, because when you're pretraining networks, you can resume where you left off in both your dataset and the pretraining.

For DataSets, you can preserialize them in an external process for later use. You can then use a [ListDataSetIterator](../doc/org/deeplearning4j/datasets/iterator/impl/ListDataSetIterator.html) to iterate over the DataSet that's been presaved. That makes it very easy to create your own custom data set iterator.

A more complete example below:

              DataSet d = DataSet.load(new File("your data"));
              //batches of 10
              DataSetIterator iter = new ListDataSetIterator(d.asList(),10);

In scaleout, you can also customize model saving with the following:

                 ActorNetworkRunner runner = ...;
                 runner.setModelSaver(...);

The default behavior is to store each snapshot of the neural net at each batch, if this is not deisrable behavior, please override this with your own custom ModelSaver.