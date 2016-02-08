---
title: Deeplearning4j Architecture
layout: default
---

# Deeplearning4j Architecture

In building Deeplearning4j, we made a series of design decisions, explained in depth below.

### Minibatch processing

To train, deep-learning nets must ingest data, and to train well, they usually need to ingest a great deal. The networks can process the input more quickly and more accurately by ingesting minibatches 5-10 elements at a time in parallel. 

This can best be illustrated with an example. Let's say you have a master coordinating 10 worker nodes. The master will shard a dataset of, say, 100 into 10 minibatches of 10, sending one minibatch to each worker. Those workers will process the input simultaneously, send their results to the master, which will average the results to create a new source of truth that is finally redistributed to the workers.

Users of DL4J's distributed deep-learning can choose the minibatch size and the software will optimally distribute the workload across workers. This is an optimal input split similar to Hadoop's. The system will maximize the number of processing units available to it, automatically scaling horizontally and vertically.

Deep-learning actually learns *better* on minibatches of 5 to 10, rather than 100. Minibatches allow the net to encounter different orientations of the data, which it then reassembles into a bigger picture.

If the batch size is too large, the network tries to learn too quickly. Smaller batch sizes force the learning to slow down, which decreases the chances of divergence as the net approaches its error minimum.

Minibatches also allow for parallel learning in the cluster, which is crucial to increasing how fast a network trains. Thus, they allow learning to remain slow even as training speeds up.

In Google's large-scale distributed training, minibatches are part of its [Sandblaster tool](http://research.google.com/archive/large_deep_networks_nips2012.html). The algorithm used to process them is called [Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS), or L-BFGS.

### Parameter averaging

Parameter averaging is an integral part of distributed training. Parameters are the weights and biases of node layers in the net. When training is distributed to several workers, they will send different sets of parameters back to the master. Parameter averaging is how the master coordinates the workers efforts, taking into account differences in the data that each worker trained on. 

Enabling parallelism by assimilating simultaneous results is just one advantage of parameter averaging. Another benefit is to prevent overfitting. In machine learning, overfitting is reading too much into the training data; i.e. it makes overly broad assumptions about data it hasn't seen based on the input you feed it. 

Parameter averaging prevents over fitting by splitting a given data set up into many subsets, none of which are learned too strongly. The net then learns the average, which is better in aggregate than the results of any one minibatch.

This tool is similar to L2 regularization, which also ensures that weights don't get too big.

It's important to note that the parameters of each layer in a multilayer network, including the output layer, are added up and averaged. 

### Job coordination

Job coordination is central to the distributed training of neural nets, the method by which many worker nodes can act in concert. Its overarching theme is internode communication. In Deeplearning4j, job coordination depends on Zookeeper and Hazelcast. 

The procedure is best described with an example: When you start your master node, it spawns local workers to enable multithreaded work. The master spins out workers according to the number of cores available. External nodes are able to connect later: the system is built of one master and *n* workers.  

Job coordination monitors the jobs assigned to each worker. Each job is associated with a minibatch. Workers, as they complete their jobs, will "heartbeat" back to the master to signal their availability. They also separately update a state tracker when their work is completed.

Job coordination, in this case, also notifies the master of job failure, in which case the master can retry. One example of job failure would be a matrix multiplication error, which will send a signal to shut down training on the worker where the error occurred. These types of errors result from misconfiguration -- i.e. the number of inputs is unequal to the columns in the data -- and it happens early. Training typically should not fail in the middle. 

Job coordination is set up with an actor network runner, which is responsible for starting the Akka cluster and connecting with the nodes in the cluster. It handles the instantiation and training of neural nets. 

Zookeeper acts as the central store for configuration; it's what the system uses to bootstrap and connect with Akka and Hazelcast. When workers start, they conduct service discovery, locate the master on a given host, and retrieve the config from that master. The configuration determines the neural network architecture.

Configuration will create a single-threaded network at each node, an actual deep-belief network or restricted Boltzmann machine. The conf contains metadata about where the master is, the absolute path to the master actor, or parameter server. The parameter server functions as the source of truth, or current best model, for all nodes. 

### Amazon Web Services Support

Deeplearning4j allows you to programmatically set up nodes. With it, you can create EC2 instances, and provision and read data from S3. 
