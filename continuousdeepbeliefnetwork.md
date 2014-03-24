---
title: 
layout: default
---

*previous* - [deep-belief networks](../deepbeliefnetwork.html)
# continuous deep-belief networks

A more extensive definition of [deep-belief networks](../deepbeliefnetwork.html) is [here](http://www.scholarpedia.org/article/Deep_belief_networks). A continuous deep-belief network is simply an extension of a deep-belief network which accepts a continuum of integers, rather than binary data.

### parameters & k

See [the parameters common to all multilayer networks](../multinetwork.html).

The variable k is the number of times you run [contrastive divergence](../glossary.html#contrastivedivergence). Each time contrastive divergence runs, it is a sample of the Markov chain. In composing a CDBN, a typical value is one.

### initiating a continuous deep-belief network

Here's how you set up a single-thread continuous deep-belief network: 

To create it, you instantiate an object of the class [CDBN](../doc/org/deeplearning4j/dbn/CDBN.html).
   
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
					{
					{0.5, 0.5, 0., 0., 0., 0.},
					{0., 0., 0., 0.5, 0.5, 0.},
					{0.5, 0.5, 0.5, 0.5, 0.5, 0.}
			});

	        DoubleMatrix predict = dbn.predict(testX);

			Evaluation eval = new Evaluation();
			eval.eval(y, predict);
			System.out.println(eval.stats());


This will print out the f1 score of the prediction.

Note that the eval class combines [confusion matrices](../glossary.html#confusionmatrix) and f1 scores to allow for easy display and evaluation of data by allowing input of outcome matrices. This is useful for tracking how well your network trains over time. 

The f1 score will be a percentage. It's basically the probability that your guess is correct. Eighty-six percent is industry standard; a solid deep-learning network should be capable of scores in the high 90s.

If you run into trouble, try modifying the hidden layer sizes, and tweaking other parameters to get the f1 score up.

Next, we'll show you how to use [distributed and multithreaded computing](../scaleout.html) to train your networks more quickly.

