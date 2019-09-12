---
title: "Deeplearning4j on Spark: Technical Explanation"
short_title: Technical Explanation
description: "Deeplearning4j on Spark: Technical Explanation"
category: Distributed Deep Learning
weight: 1
---

# DL4J Distributed Training: Technical Explanation

This section will cover the technical details of Deeplearning4j's Apache Spark gradient sharing training implementation. Details on the parameter averaging implementation also follow.  Note that the parameter averaging implementation has been superseded by the gradient sharing implementation as of 1.0.0-beta. This guide assumes the reader is familiar with key concepts in distributed training like data parallelism and synchronous vs asynchronous SGD. This [blog post](https://blog.skymind.ai/distributed-deep-learning-part-1-an-introduction-to-distributed-training-of-neural-networks/) can provide an introduction.

* [Asynchronous SGD Implementation](#asgd)
* [Parameter Averaging Implementation](#parameteravg)
* [Fault Tolerance](#faulttol)

## <a name="asgd">Asynchronous SGD Implementation</a>
DL4J's asynchronous SGD implementation is based on the [Strom 2015 neural network training paper](http://nikkostrom.com/publications/interspeech2015/strom_interspeech2015.pdf) by Nikko Strom, with some modifications.
The next section will review the key features of the Strom paper followed by another section that describes the DL4J implementation and how it differs from the paper.

### Strom's Approach
When training a neural network on a cluster, the worker machines need to communicate changes to their parameters - either by communicating the new parameter values directly (such as in parameter averaging) or by communicating gradient/update information (as in gradient sharing).

The key feature of this approach is that opposed to relaying all parameters/updates across the network only updates that are above a user specified threshold are communicated. Put another way: we start out with an update vector (1 entry per parameter) that needs to be communicated. Instead of communicating the vector as-is, we communicate only the large elements in a quantized way (which is a sparse binary vector) instead of all elements.
The motivation here is to reduce the amount of network communication required - this "sparse, 1-bit binary encoding" approach can reduce the size required for communicating updates by a factor of 1000x or more - see the Strom paper for some compression statistics.

Note that updates below the threshold are not discarded but accumulated in a “residual” vector to be applied later. Also of note is the absence of a centralized parameter server which is replaced by peer to peer communication as indicated in the image below.

![Strom's ASGD implementation](/images/guide/Strom_ASGD.svg)

The update vectors, δi,j in the image above, are:
1. Sparse: only some of the gradients are communicated in each vector δi,j (the remainder are assumed to be 0) - sparse entries are encoded using an integer index
2. Quantized to a single bit: each element of the sparse update vector takes value +τ or −τ. This value of τ is the same for all elements of the vector, hence only a single bit is required to differentiate between the two options
3. Integer indexes (used to identify the entries in the sparse array) are optionally compressed using entropy coding to further reduce update sizes (the author quotes a further 3x reduction at the cost of additional computation, though the benefit may not be worth the additional cost)

One of the main concerns of asynchronous SGD is the issue of stale gradients. Stale gradients need not be explicitly handled in Strom's approach - in most cases, the updates are applied very quickly on each node. The paper reports a reduction in network transfers by several orders of magnitude. Given a suitably computation intensive model (like an RNN or a CNN) this drastic reduction in network communication ensures that model equivalency is maintained across all nodes and stale gradients are not an issue.

However the approach is not without its downsides as described below:
1. Strom reports that convergence can suffer in the early stages of training (using fewer compute nodes for a fraction of an epoch seems to help)
2. Compression and quantization is not free: these processes result in extra computation time per minibatch, and a small amount of memory overhead per executor
3. The process introduces two additional hyperparameters to consider: the value for the threshold, τ and whether to use entropy coding for the updates or not (though notably both parameter averaging and async SGD also introduce additional hyperparameters)


### DL4J's ASGD implementation

The DL4J implementation differs from Strom's approach in the following ways:

1. Not point-to-point: 
The implementation allows the user to choose between two modes of network organization - plain mode and mesh mode. Plain mode is to be used when the number of nodes in the cluster are < 32 nodes and mesh mode is to be used for larger clusters. Refer to the section on [different modes](#modes) for more details.
2. Two encoding schemes:
	DL4J uses two encoding schemes, dynamically switching between the two depending on which will provide less network communication. Refer to the section on [encoding](#encoding) for more details.
3. Quantization thresholds adjusted:
	The quantization threshold is stepped up or down depending on the distribution of the updates after each iteration. This is done on each node independently to make sure that updates are indeed sparse. In practice, this is implemented via the [ThresholdAlgorithm](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/solvers/accumulation/encoding/ThresholdAlgorithm.java) interface and the [implementations](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/solvers/accumulation/encoding/threshold) there-of.
4. Residual clipping
	As noted earlier, the "left over" parts of the updates (i.e., those parts not communicated) are store in the residual vector. If the updates are much larger than the threshold, we can have a phenomenon we have termed "residual explosion" - that is, the residual values can continue to grow to many times the threshold (hence would take many steps to communicate the gradient). To avoid this, DL4J has a [ResidualPostProcessor](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/solvers/accumulation/encoding/ResidualPostProcessor.java) interface, with the default implementation being [ResidualClippingPostProcessor](https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/solvers/accumulation/encoding/residual/ResidualClippingPostProcessor.java) which clips the residual vector to a maximum of 5x the current threshold, every 5 steps.
5. Local parallelism via ParallelWrapper: 
  This enables multi-CPU/GPU nodes to share information faster


As is evident from the description, an implementation of ASGD requires updates to be transferred with every iteration of training. Further communication between workers within the cluster is a requirement in mesh mode.

To enable fast out of spark communication DL4J uses [Aeron](https://github.com/real-logic/aeron/wiki). Aeron is a high performance messaging system that can run over UDP, Infiniband or Shared Memory. Aeron is designed to be the highest throughput with the lowest and most predictable latency possible of any messaging system. Building our own communications stack above Aeron allows us to have a custom implementation of the parameter server integrated with Spark and yet control and minimize allocations right of the wire.

#### <a name="modes">Plain Mode vs Mesh Mode</a>

DL4J's gradient sharing implementation can be configured in 2 ways, depending on the cluster size.

Below is an image describing how plain mode is organized:
![Plain Mode](/images/guide/plainmode.png)


In plain mode, quantized encoded updates are relayed by each node to the master and the master then relays them to the remaining nodes. This ensures that the master always has an up to date version of the model, which is necessary for fault tolerance. The master node however is a potential bottleneck in this implementation. To scale to larger sized cluster (more than about 32 nodes - though this is network and hardware specific) use mesh mode as described below.

Below is an image describing how mesh mode is organized:
![Mesh Mode](/images/guide/meshmode.png)

Mesh mode is a non-binary tree with Spark master at its root. By default each node can have a maximum of eight nodes and the tree can be a maximum of five levels deep. In mesh mode each node relays encoded updates to all nodes connected to it and each node aggregates updates received from all other nodes connected to it. In mesh mode, the master is no longer a bottleneck as the amount of communication it recieves directly is reduced. As the writing of this document, the implementation has been tested with unicast as well as multicast (available in 1.0.0-beta3). Future support is planned for RDMA.

#### <a name="encoding">Encoding Schemes</a>
Updates are send using one of two schemes as described below.
  * Threshold encoding: Sends an array of integers each referring to the index of the parameter. A positive integer is send for a positive threshold and a negative integer is send for a negative threshold.
  * Bitmap encoding: Each parameter update is encoded with two bits. The four states are used to indicate no change, a +ve threshold change, a -ve threshold change and a half threshold change that cycles between +ve and -ve.

Using these two kinds of encoding schemes accommodates cases when the updates are dense. Since each node has its own threshold it's value is also communicated with each transfer. Encoding updates are pushed down to optimized native code (c++) for the sake of performance and GPU parallelization.
The sparse threshold (integer index) encoding can result in very high compression rates, whereas the bitmap encoding results in a fixed size 16x compression ratio (i.e., 2 bits per parameter vs. 32 bits for the original update vector).


## <a name="parameteravg">Parameter Averaging Implementation</a>
The parameter averaging implementation was the first distributed training implementation in DL4J. It has since been superseded by the gradient sharing implementation described in the previous section. Details on the parameter averaging implementation are included here for the sake of completeness.

The parameter averaging implementation is a synchronous SGD approach implemented entirely in Spark. DL4J's parameter averaging implementation uses a single parameter server, a role served by the Spark master node. 

Parameter averaging is the conceptually simplest approach to data parallelism. It requires the user to specify the frequency at which the workers synchronize with each other and the master. With parameter averaging, training proceeds as follows:

1. The master (Spark driver) starts with an initial network configuration and parameters
2. Data is split into a number of subsets, based on the configuration of the TrainingMaster.
3. Iterate over the data splits. For each split of the training data:
  a. Distribute the configuration, parameters (and if applicable, network updater state for momentum/rmsprop/adagrad) from the master to each worker
  b. Fit each worker on its portion of the split
  c. Average the parameters (and if applicable, updater state) and return the averaged results to the master
4. Training is complete, with the master having a copy of the trained network

Steps 3a through 3c are demonstrated in the image below. In this diagram, W represents the parameters (weights, biases) in the neural network. Subscripts are used to index the version of the parameters over time, and where necessary for each worker machine.

![Parameter Averaging](/images/guide/parameteraveraging.svg)

The implementation uses Spark's treeAggregate under the hood. There are a number of enhancements that can be made to this implementation that will result in faster training times. Even with these enhancements in place the asynchronous SGD approach with quantized compressed updates is expected to continue to be much faster. Therefore the user is strongly recommended to switch from the parameter averaging implementation to the asynchronous SGD gradient sharing approach.


## <a name="faulttol">Fault Tolerance</a>

Spark implementations of distributed training in DL4J are fault tolerant as of 1.0.0-beta3.
The parameter averaging implementation has always been fault tolerant; the gradient sharing implementation was made fully fault tolerant after (not including) 1.0.0-beta2.

Before going into the details of the implementation let us first consider what happens when a node goes down. Since Spark is unaware of the updates send via Aeron the RDD lineage tracks back to the initial parameter and optimizer state. When Spark restores a node in place of one that went down it will therefore will resume training from its initial state. In other words, this restored node will be out of sync with the other nodes and this will cause training to diverge.

DL4J's Gradient sharing utilizes its own internal heartbeat mechanism outside of Spark to detect when a node goes down, as well as to detect when a recovered node comes online. To ensure that training continues without diverging it is necessary that the restored node resumes training with a copy of the model identical to that on the other nodes at the current point. To ensure that updates are not applied multiple times each update is tagged with a unique ID. The state of the updater/optimizer (RMSProp, AdaGrad etc) as well as the iteration/epoch number are also required for network training to proceed from the state prior to the node failure. 

The following outlines what happens when a node goes down in plain mode and is restored:
1. The restored node reconnects to the master node
2. The restored node starts receiving updates and then sends request for parameters, updater state and current epoch/iteration 
3. Master fulfils these requests (by itself or by proxy)
4. The restored node applies ONLY relevant updates (relative to the parameter vector)
4. Training continues on the RDD data on the new node, properly in-sync with other nodes and properly converging

Requesting a copy of the model after the node has started receiving updates makes sure that updates are not missed. Updates are tagged by unique IDs and no update will be incorrectly applied twice. Since the master does not do any training it does not hold the updater state, when it receives a request for the updater/optimizer state it sends out a request to one of the other nodes - upon receiving the request, it sends the updater to the restored node.

The only additional step in mesh node when a node fails is to remap the descendants of the failed node. In this case a descendant of the failed node is mapped to master and all the remaining descendants are mapped to the one mapped to master.

Concretely with the tree structure below if node 2 fails, node 5 is mapped to the master and node 6 and 7 are mapped to node 5.

![Node Failure](/images/guide/nodefailure.png)


The decision to remap to master instead of the neighboring nodes was made since the master is assumed to be the most reliable option. Requesting a copy of the model etc are also made to the master for this very same reason. It is to be noted that similar to a Spark job distributed neural network training with DL4J cannot withstand the master node failing. For this reason, the user is advised to persist the state of the model frequently. In this case if the master were to fail training can be restarted from the latest saved state. 

Limitations of fault tolerance: There are two main limitations of fault tolerance for the gradient sharing implementation.
First: A small amount of data (a few minibatches) may be processed multiple times. This is because a failed node may process part of a partition (sending out updates) before failing. This is not a problem in practice: the number of duplicated minibatches is usually very small, and we are typically training for multiple epochs anyway (thus each example is already being seen multiple times during training).
Second: The master/driver node is a single point of failure. This is essentially a Spark limitation: DL4J could (in principle) implement functionality to recover from a failed master and continue training, but Apache Spark does not support fault tolerance for the master node.

