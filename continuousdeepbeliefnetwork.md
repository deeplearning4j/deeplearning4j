---
title: 
layout: default
---

# continuous deep-belief network



For our purposes, a [deep-belief network](http://www.scholarpedia.org/article/Deep_belief_networks) can be defined as a stack of restricted Boltzmann machines in which each layer communicates with its previous and subsequent layers, while the nodes of any single layer do not communicate with each other laterally. With the exception of the  first and final layers, each hidden layer has a double role: it serves as the hidden layer to the higher nodes before, and as the input layer to the lower nodes after. It is a network of networks. 

Deep-belief networks are used to recognize and generate images, video sequences and motion-capture data. 


A continuous deep belief network is an extension of a deep belief network that accepts float input.


### parameters

Please also see [the multilayer network parameters common to all multilayer networks]({{ site.baseurl }}/multinetwork.html)

#### k 

K is the number of times you run [contrastive divergence]({{ site.baseurl }}/glossary.html#contrastivedivergence). Each time contrastive divergence is run, it is a sample of the Markov chain. In composing the restricted Boltzmann machine, a typical value is one.

### initiating a deep-belief network

Setting up a single-thread deep belief network is easy. 

To create the machine, you simply instantiate an object of the [class]({{ site.baseurl }}/doc/com/ccc/deeplearning/dbn/CDBN.html).

    
   //training data and labels
   DoubleMatrix x = new DoubleMatrix(new double[][] {

				{ 0.4, 0.5, 0.5, 0,  0,  0 },
				{0.5, 0.3,  0.5, 0.,  0.,  0.},
				{0.4, 0.5, 0.5, 0.,  0.,  0.},
				{0.,  0.,  0.5, 0.3, 0.5, 0.},
				{0.,  0.,  0.5, 0.4, 0.5, 0.},
				{0.,  0.,  0.5, 0.5, 0.5, 0.}
				
		});

		DoubleMatrix  y = new DoubleMatrix(new double[][]
				{

			   {1, 0},
				{1, 0},
				{1, 0},
				{0, 1},
				{0, 1},
				{0, 1}
				});

		RandomGenerator rng = new MersenneTwister(123);

		double preTrainLr = 0.01;
		int preTrainEpochs = 1000;
		int k = 1;
		int nIns = 6,nOuts = 2;
		int[] hiddenLayerSizes = new int[] {5,5};
		double fineTuneLr = 0.01;
		int fineTuneEpochs = 200;

        //Initialization
		CDBN dbn = new CDBN.Builder()
		.numberOfInputs(nIns).numberOfOutPuts(nOuts)
		.hiddenLayerSizes(hiddenLayerSizes).useRegularization(false)
		.withRng(rng).withL2(0.1).renderWeights(1000)
		.build();
		
		//Train the network
		dbn.pretrain(x,k, preTrainLr, preTrainEpochs);
		dbn.finetune(y,fineTuneLr, fineTuneEpochs);


		DoubleMatrix testX = new DoubleMatrix(new double[][]
				{{0.5, 0.5, 0., 0., 0., 0.},
				{0., 0., 0., 0.5, 0.5, 0.},
				{0.5, 0.5, 0.5, 0.5, 0.5, 0.}});

     DoubleMatrix predict = dbn.predict(testX);

		Evaluation eval = new Evaluation();
		eval.eval(y, predict);
		System.out.println(eval.stats());


This will print out the f score of the prediction. 

The eval class combined confusion matrices and f1 scores to allow for easy display and evaluation of data

by allowing input of outcome matrices. This is useful for tracking how well your network is training over time.

The f1 score will be a percentage similar to a probability of being correct: (86%+ is good.)


If there's issues, try modifying the hidden layer sizes, and tweaking other parameters to work on 

getting your f1 score up.

