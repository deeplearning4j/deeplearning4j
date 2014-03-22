---
title: 
layout: default
---

# restricted Boltzmann machine

To quote Hinton, a [Boltzmann machine](http://www.scholarpedia.org/article/Boltzmann_machine) is "a network of symmetrically connected, neuron-like units that make stochastic decisions about whether to be on or off." 

A [restricted Boltzmann machine](http://www.scholarpedia.org/article/Boltzmann_machine#Restricted_Boltzmann_machines) "consists of a layer of visible units and a layer of hidden units with no visible-visible or hidden-hidden connections." That is, its nodes must form a symmetrical [bipartite graph](https://en.wikipedia.org/wiki/Bipartite_graph). 

![Alt text](../img/bipartite_graph.png)

A trained RBM will learn the structure of the data fed into it; it does so through the very act of reconstructing the data again and again. 

[RBMs](../glossary.html#restrictedboltzmannmachine) are useful for [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction), [classification](https://en.wikipedia.org/wiki/Statistical_classification), [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering), [feature learning](https://en.wikipedia.org/wiki/Feature_learning) and [topic modeling](https://en.wikipedia.org/wiki/Topic_model). Given their relative simplicity, they're the first neural network we'll tackle.

### parameters

See [the parameters common to all single-layer networks](../singlelayernetwork.html).

### k 

The variable k is the number of times you run [contrastive divergence](../glossary.html#contrastivedivergence). Each time contrastive divergence is run, it's a sample of the Markov chain composing the restricted Boltzmann machine. A typical value is 1.

## initiating an RBM

Here's how you set up a single-thread restricted Boltzmann machine: 

To create it, you simply instantiate an object of the [RBM class](../doc/org/deeplearning4j/rbm/RBM.html).


		   RBM rbm = new RBM.Builder().numberOfVisible(784).numHidden(400).withRandom(rand)
				.useRegularization(false).withMomentum(0).build();

The RBM uses the builder pattern to set up config; for example, this builder will handle the following parameters:

           Number of visible (input) units: 784

           Number of hidden (output) units: 400 

           withRandom(specify an RNG)

           useRegularization(use L2?)

           Momentum: Use momentum or not?

Next, create a training set for the machine. For the sake of visual brevity, a toy, two-dimensional data set is included in the code below. (With large-scale projects, training sets are clearly much larger.)


            double[][] data = new double[][]
				{
				{1,1,1,0,0,0},
				{1,0,1,0,0,0},
				{1,1,1,0,0,0},
				{0,0,1,1,1,0},
				{0,0,1,1,0,0},
				{0,0,1,1,1,0},
				{0,0,1,1,1,0}
			};

		    DoubleMatrix d = new DoubleMatrix(data);

Now that you've instantiated the machine and created the training set, it's time to train the network. This will run contrastive divergence until convergence, with a learning rate of 0.01, a k of 1 and with the input specified above. 

The last snippet will construct a new training set and show the reconstructed input. Note that RBMs only take binary input, not a continuum of integers.


		  rbm.trainTillConvergence(0.01,1,d);
		
          double[][] testData = new double[][]
			  {
			    {1, 1, 0, 0, 0, 0},
				{0, 0, 0, 1, 1, 0}
			  };

		  DoubleMatrix v = new DoubleMatrix(testData);	

          System.out.println(r.reconstruct(v).toString());

Once built, you can test your trained network by feeding it unstructured data and checking the output.

You can intrepret the output numbers as percentages. Every time the number in the reconstruct is *not zero*, that's a good indication the network learned the input. We'll have a better example later in the tutorials.

Next, we'll show you how to implement a [Deep-Belief Network](../deepbeliefnetwork.html), which is simply many RBMs strung together.