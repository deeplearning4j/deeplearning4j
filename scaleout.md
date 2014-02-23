---
title: 
layout: default
---


# scaleout: use iterative reduce on multithreaded training

Training a neural network is time consuming without some kind of parallelism. The scaleout module in deeplearning4j-scalout uses akka for both clustering for distributed computing as well as multithreaded computing.

Here's a snippet for training a network:




        DataSetIterator iter = ...;
		//run the network runner as master (worker is used for anything that is just a supporting role for more computing power)
		ActorNetworkRunner runner = new ActorNetworkRunner("master", iter);
		//you only need to create the conf for master.
		Conf conf = new Conf();
		conf.setFinetuneEpochs(1000);
		conf.setPretrainLearningRate(0.01);
		conf.setLayerSizes(new int[]{1000 ,500,250});
		conf.setMomentum(0);
		//pick the multi layer network to train: note that this is the MULTI layer
		conf.setMultiLayerClazz(CDBN.class);
		conf.setnOut(d1.getSecond().columns);
		conf.setFunction(new HardTanh());
		conf.setFinetuneLearningRate(0.01);
		conf.setnIn(d1.getFirst().columns);
		conf.setMomentum(0.9);
		conf.setUseRegularization(false);
		//always call setup
		runner.setup(conf);
		//train the network using the passed in data set iterator.
		runner.train();




### ZooKeeper

This relies on having a zookeeper instance setup -- otherwise the system will stall.

### ActorNetworkRunner

There are two different ActorNetworkRunners: one for single and another for multi. Both have similar APIs so they're easy to use, but that can also lead to confusion.

If your cluster does not start after calling train, this is likely a race condition of sending the data for training relative to the cluster starting. Submit a pull request if that's the case.

After training, all models are saved in the same directory where the user started. These models will be named nn-model-*.bin. That's the output of the network. (These neural networks are parameter averaged.)

### Roles

*Master*: Used for multithreading and for seeding a cluster.

*Worker*: Used for connecting to a master network runner for lending cpu power.


### Conf

The configuration class has a lot of knobs. It handles both the single-layer network and the multilayer networks. Call multiLayerClazz or setNeuralNetworkClazz, respectively. 

