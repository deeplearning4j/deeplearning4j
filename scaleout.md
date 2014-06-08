---
title: 
layout: default
---

*previous* - [deep-belief networks](../deepbeliefnetwork.html)
# scaleout: iterative reduce on multithreaded training

Training a neural network is very time consuming without some kind of parallelism. The scaleout module in deeplearning4j-scaleout uses akka for both clustering for distributed computing as well as multithreaded computing.

Here's a snippet for training a network:

<script src="http://gist-it.appspot.com/github.com/agibsonccc/java-deeplearning/blob/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/mnist/MnistExampleMultiThreaded.java?slice=30:53"></script>

Note that runner.train() is not a blocking call.

### ZooKeeper

This relies on having a zookeeper instance setup -- otherwise the system will stall.

### ActorNetworkRunner

There are two different ActorNetworkRunners: one for single and another for multi. Both have similar APIs so they're easy to use, but that can also lead to confusion.

If your cluster does not start after calling train, this is likely a race condition of sending the data for training relative to the cluster starting. Submit a pull request if that's the case.

After training, all models are saved in the same directory where the user started. These models will be named nn-model-*.bin. That's the output of the network. (These neural networks are parameter averaged.)

### Roles

*Master*: Used for multithreading and seeding a cluster.

*Worker*: Used for connecting to a master network runner for lending cpu power.

### Conf

The configuration class has a lot of knobs. It handles both the single-layer network and the multilayer networks. Call multiLayerClazz or setNeuralNetworkClazz, respectively. 

### Costs

Setting up a cluster on [EC2](https://aws.amazon.com/ec2/) costs anywhere from a few cents to a few dollars an hour. While this may be prohibitive for some, the intense computational needs of deep learning make parallelism necessary for any serious work. 

Next, we'll show you how to [run a worker node](../distributed.html).