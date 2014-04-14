---
title: 
layout: default
---

*previous* - [restricted Boltzmann machines](../restrictedboltzmannmachine.html)
# continuous restricted boltzmanns

A continuous restricted Boltzmann machine is a form of RBM that accepts continuous input (i.e. numbers cut finer than integers) via a different type of contrastive divergence sampling. This allows it to handle things like image pixels or word-count vectors that are normalized to decimals between zero and one.

It should be noted that every layer of a deep-learning net needs four elements to work: the input, the coefficients, the bias and the transformation. 

The input is the numeric data, a vector, fed to it from the previous layer. The coefficients are the weights given to various features passing through the node layer. The bias ensures that some nodes in a layer will be activated no matter what. The transformation is an additional algorithm that processes data after it passes through each layer. 

Those additional algorithms and their combinations layer by layer can vary. The most effective CRBM we've found employs a Gaussian transformation on the visible, or input, layer, and a rectified-linear-unit tranformation on the hidden layer. We've found this particularly useful in [facial reconstruction](../facial-reconstruction-tutorial.html). 

*A brief aside on transformations: We've found that Gaussian transformations do not work well on hidden layers, which are where the reconstructions happen; i.e. the layer that matters. Rectified-linear-unit transformations are capable of representing more features than binary transformations do.*

### parameters 

See also [the parameters common to all single-layer networks](../singlelayernetwork.html).

### k

The variable k is number of times you run [contrastive divergence](../glossary.html#contrastivedivergence). Each time contrastive divergence is run, it is a sample of the Markov chain. Composing the restricted Boltzmann machine, a typical value for k is 1.

### initiating a CRBM

Setting up a single-thread continuous restricted Boltzmann machine is easy. To create the machine, you simply instantiate an object of the class [CRBM](../doc/org/deeplearning4j/rbm/CRBM.html).
    
     DoubleMatrix input = new DoubleMatrix(new double[][]{
				{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.5, 0.3,  0.5, 0.,  0.,  0.},
				{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.,  0.,  0.5, 0.3, 0.5, 0.},
				{0.,  0.,  0.5, 0.4, 0.5, 0.},
				{0.,  0.,  0.5, 0.5, 0.5, 0.}});

	  CRBM r = new CRBM.Builder().numberOfVisible(input.getRow(0).columns).numHidden(10).build();


This created a continuous restricted Boltzmann machine with the number of inputs matching the number of columns in the input. The number of outputs are 10.

Next, create a training set for the machine. For the sake of visual brevity, a toy, two-dimensional data set is included in the code below. (With large-scale projects, training sets are 
clearly much more substantial.)

      double[][] data = new double[][] {
				{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.5, 0.3,  0.5, 0.,  0.,  0.},
				{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.,  0.,  0.5, 0.3, 0.5, 0.},
				{0.,  0.,  0.5, 0.4, 0.5, 0.},
				{0.,  0.,  0.5, 0.5, 0.5, 0.}
			};

       DoubleMatrix input = new DoubleMatrix(data);

Now that you have instantiated the machine and created the training set, it's time to train the network. 

		rbm.trainTillConvergence(0.01,1,input);

This trains the CRBM until convergence with a learning rate of 0.01, a k of 1, and the specified input. You can test your trained network by feeding it unstructured data and checking the output. 

     
     double[][] data = new double[][] {
				{0.5, 0.5, 0., 0., 0., 0.},
				{0., 0., 0., 0.5, 0.5, 0.}
 
	 };

     DoubleMatrix test = new DoubleMatrix(data);

     System.out.println(r.reconstruct(test).toString());