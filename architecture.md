---
title: 
layout: default
---

# architecture

In building Deeplearning4j, we made a series of design decisions explained in depth below.

### minibatch processing

To train, deep-learning nets must ingest data, and to train well, they usually need to ingest a great deal. The networks can process the input more quickly and more accurately by ingesting minibatches 5-10 elements at a time in parallel. 

This can best be illustrated with an example. Let's say you have a master coordinating 10 worker nodes. The master will shard a dataset of, say, 100 into 10 minibatches of 10, sending one minibatch to each worker. Those workers will process the input simultaneously, send their results to the master, which will average the results to create a new source of truth that is finally redistributed to the workers.

Users of DL4J's distributed deep-learning can choose the minibatch size and the software will optimally distribute the workload across workers. This is an optimal input split similar to Hadoop's. The system will maximize the number of processing units available to it: it automatically scales horizontally and vertically.

Deep-learning actually learns *better* on minibatches of 5 to 10, rather than 100. Minibatches allow the net to encounter different orientations of the data, which it then reassembles into a bigger picture.

If the batch size is too large, the network tries to learn too quickly. Smaller batch sizes force the learning to slow down, which decreases the chances of divergence as the net approaches its error minimum.

Minibatches also allow for parallel learning in the cluster, which is crucial to increasing how fast a network trains. Thus, they allow learning to remain slow even as training speeds up.

In Google's large-scale distributed training, minibatches are part of its [Sandblaster tool](http://research.google.com/archive/large_deep_networks_nips2012.html). The algorithm used to process them is called [Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS), or L-BFGS.