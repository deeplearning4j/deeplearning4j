---
title: 
layout: default
---


Scaleout - Use Iterative Reduce on multi threaded training
=====================================================================


Training a neural network is often time consuming without some kind of parallelism.

The scaleout module in deeplearning4j-scalout uses akka for both clustering for distributed computing

as well as multithreaded computing.


An example snippet on training a network is the following:




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




===ZooKeeper

    Note that this relies on having a zookeeper instance setup. Otherwise the system will stall.


Nuances
=============================



ActorNetworkRunner

     Note that there are two different ActorNetworkRunners. One meant for single and another meant for multi.
     
     Both have similar apis for ease of use, but this can also lead to confusion.



Roles

    Master - Used for multithreading and for seeding a cluster.

    Worker - Used for connecting to a master network runner for lending cpu power.




Conf

    The configuration class has a lot of knobs.

    Keep in mind that this conf handles both the single layer network and the multi layer networks.

    Call multiLayerClazz or setNeuralNetworkClazz respectively. 

