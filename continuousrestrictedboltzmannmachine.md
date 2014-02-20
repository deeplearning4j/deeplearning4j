---
title: 
layout: default
---

# continuous restricted boltzmann

A continuous restricted Boltzmann machine is a form of RBM that accepts continuous input via a different type of contrastive divergence sampling. This allows it to handle things like image pixels or word count vectors that are normalized to probabilities.

### parameters 

Please also see [the single-layer network parameters common to all single-layer networks]({{ site.baseurl }}/singlelayernetwork.html).

### k

K is number of times you run [contrastive divergence]({{ site.baseurl }}/glossary.html#contrastivedivergence). Each time contrastive divergence is run, it is a sample of the Markov chain.

Composing the restricted Boltzmann machine. A typical value of 1 is fine.

### initiating a continuous restricted Boltzmann machine

Setting up a single-thread continuous restricted Boltzmann machine is easy. 

To create the machine, you simply instantiate an object of the [class]({{ site.baseurl }}/doc/com/ccc/deeplearning/rbm/CRBM.html).
    

     DoubleMatrix input = new DoubleMatrix(new double[][]{
				{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.5, 0.3,  0.5, 0.,  0.,  0.},
				{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.,  0.,  0.5, 0.3, 0.5, 0.},
				{0.,  0.,  0.5, 0.4, 0.5, 0.},
				{0.,  0.,  0.5, 0.5, 0.5, 0.}});

	  

	  CRBM r = new CRBM.Builder().numberOfVisible(input.getRow(0).columns).numHidden(10).build();



This created a continuous restricted boltzmann machine with the number of inputs matching the number of columsn in the input.
The number of outputs are 10.


Next, create a training set for the machine. For the sake of visual brevity, a toy, two-dimensional data set is included in the code below. (With large-scale projects, training sets are clearly much more substantial.)

 DoubleMatrix input = new DoubleMatrix(new double[][]{
				{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.5, 0.3,  0.5, 0.,  0.,  0.},
				{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.,  0.,  0.5, 0.3, 0.5, 0.},
				{0.,  0.,  0.5, 0.4, 0.5, 0.},
				{0.,  0.,  0.5, 0.5, 0.5, 0.}});

Now that you have instantiated the machine and created the training set, it's time to train the network. 

		rbm.trainTillConvergence(0.01,1,input);



You can test your trained network by feeding it unstructured data and checking the output. 




 DoubleMatrix test = new DoubleMatrix(new double[][]
				{{0.5, 0.5, 0., 0., 0., 0.},
				{0., 0., 0., 0.5, 0.5, 0.}});

 System.out.println(r.reconstruct(test).toString());